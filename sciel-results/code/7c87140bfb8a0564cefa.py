import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.dim = dim
        self.gamma = 0.90
        self.epsilon = 0.1
        # Initialize EMAs to 0.0 or a small positive value to encourage initial exploration?
        # 0.0 is standard.
        self.ema_rewards = np.zeros(self.n)
        self.counts = np.zeros(self.n)

    def select(self, context):
        # Ignore context
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n))
        else:
            return int(np.argmax(self.ema_rewards))

    def update(self, context, arm, reward):
        # Ignore context
        self.counts[arm] += 1
        self.ema_rewards[arm] = self.gamma * self.ema_rewards[arm] + (1.0 - self.gamma) * reward
