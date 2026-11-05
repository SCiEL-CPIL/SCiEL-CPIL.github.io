import numpy as np


class Policy:
        def __init__(self, n_arms, dim):
            self.n = n_arms
            self.gamma = 0.90
            self.alpha = 1.0
            self.ema_mean = np.zeros(n_arms)
            self.ema_sq = np.zeros(n_arms)
            self.counts = np.zeros(n_arms) # To handle initial high uncertainty?
            
        def select(self, context):
            # Calculate variance
            var = self.ema_sq - self.ema_mean**2
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            # UCB scores
            # For arms not played, we want high exploration.
            # If counts is 0, var is 0, std is 0. Score is 0.
            # This might not explore enough initially.
            # Let's add a bonus for unplayed arms or use a different init.
            
            # Better init: Set initial mean to 0, but give a large initial uncertainty?
            # Or just use Epsilon Greedy?
            
            # Let's stick to UCB but handle the start.
            # If we initialize ema_sq to 1.0 (assuming unit variance), initial std is 1.
            # Score = 0 + 1.0 * 1.0 = 1.0.
            # After one play with reward 0:
            # mean = 0.1*0 = 0
            # sq = 0.9*1 + 0.1*0 = 0.9
            # var = 0.9 - 0 = 0.9, std ~ 0.95
            # Score ~ 0.95.
            # This encourages exploration initially.
            
            scores = self.ema_mean + self.alpha * std
            return int(np.argmax(scores))

        def update(self, context, arm, reward):
            r = reward
            self.ema_mean[arm] = self.gamma * self.ema_mean[arm] + (1 - self.gamma) * r
            self.ema_sq[arm] = self.gamma * self.ema_sq[arm] + (1 - self.gamma) * r**2
