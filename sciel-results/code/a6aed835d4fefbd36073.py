import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.dim = dim
        self.gamma = 0.90
        self.ema_rewards = np.zeros(n_arms)
        self.epsilon = 0.1
        
    def select(self, context):
        # Context is ignored as per directive
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n))
        else:
            return int(np.argmax(self.ema_rewards))
            
    def update(self, context, arm, reward):
        # Update EMA for the selected arm
        self.ema_rewards[arm] = self.gamma * self.ema_rewards[arm] + (1 - self.gamma) * reward
