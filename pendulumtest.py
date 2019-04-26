'''
Use DQN to  make the pendulum swing up.
Reward will be like sin map.
'''



import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
env = gym.make('Pendulum-v0')
N_STATES = env.observation_space.shape[0]
HIDDEN_SIZE = 15
N_ACTIONS = 3 #action 9 data shape(1,)
LR = 0.01
GAMMA = 0.9
EPSILON = 0.9
MEMORY_CAPACITY = 20
TARGET_REPLACE_ITER = 10
BATCH_SIZE = 32
env.unwrapped

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

def linear_func(theta,x):
    return theta[0,0]*x**2+theta[0,1]*x+theta[0,2]
    
class DQN(object):
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()

        self.learn_step_counter =0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        theta = self.eval_net.forward(x).data.numpy()
        if np.random.uniform() < EPSILON:# 贪心算法
            extremum_point = -theta[0,1]/(theta[0,0]*2)
            if extremum_point > -2. and extremum_point <2.: 
                # net 返回的是表达cost 和 action 的参数
                # 在 linear_func中找cost的最值，从而索引action
                list = np.array([linear_func(theta,extremum_point),
                linear_func(theta,-2.),
                linear_func(theta,2.)])
                pos = np.array([extremum_point,-2.,2.])
                action=pos[np.where(list==max(list))]
            else:
                list = np.array([
                linear_func(theta,-2.),
                linear_func(theta,2.)])
                pos = np.array([-2.,2.])
                action=pos[np.where(list==max(list))]
            action = action.reshape((1,))
        else:
            action = np.array([np.random.random_sample()*4-2])
        print("action:",action,"|predict reward:",linear_func(theta,action)*16)
        # print("theta ",theta)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        print("learn time","*"*30,self.learn_step_counter)
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])# [cos1,sin1,thetadot1],[]...
        b_a = torch.FloatTensor(b_memory[:, N_STATES:N_STATES+1].astype(float))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        q_eval = linear_func(self.eval_net.forward(b_s).data.numpy(),b_a)  # shape (batch, 1)
        q_eval.requires_grad_(True)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
cost = np.array(0)
for i_episode in range(100):
    state = env.reset()
    ep_r = 0
    while True:
        env.render()
        action = dqn.choose_action(state)

        # take action
        next_state, reward, done, info = env.step(action)

        # modify the reward
        cos,sin,thetadot = next_state
        # r = np.random.uniform()*16
        print("thetadot ",thetadot)
        # if abs(thetadot)>7.5:
        #     done = True
        cost=np.append(cost,reward)
        dqn.store_transition(state, action, reward, next_state)

        ep_r += reward
        state = next_state
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break


x = np.arange(cost.size)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, cost, 'b', label='Prediction')
ax.legend(loc=2) # 2表示在左上角
ax.set_xlabel('act')
ax.set_ylabel('reward')
plt.show()