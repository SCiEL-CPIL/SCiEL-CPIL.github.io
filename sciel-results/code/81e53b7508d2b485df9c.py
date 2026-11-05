"""Rewritten program for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Context-free multi-armed bandit with:
    - Sliding Window UCB for non-stationary rewards
    - Maintains a fixed-size window of recent rewards per arm
    - Computes mean and std from the window for UCB scores
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Sliding window size
        self.window_size = 50

        # Per-arm reward history
        self.rewards = [deque(maxlen=self.window_size) for _ in range(self.n)]

        # Exploration parameter
        self.c = 1.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 118.0
        self.round = 0

        # Fallback arm tracking
        self.best_arm = 0
        self.best_reward = -np.inf

    def _check_deadline(self):
        if time.time() - self.start_time >= self.deadline:
            return True
        return False

    def select(self, context):
        self.round += 1

        if self._check_deadline():
            return self.best_arm

        scores = np.zeros(self.n)
        for a in range(self.n):
            if len(self.rewards[a]) == 0:
                # Force exploration of unvisited arms
                scores[a] = np.inf
            else:
                mean = np.mean(self.rewards[a])
                # Use sample std with ddof=1 if possible, else population std
                if len(self.rewards[a]) > 1:
                    std = np.std(self.rewards[a], ddof=1)
                else:
                    std = 0.0
                
                # UCB1-style bonus
                # We use the global round count for the log term to ensure exploration decays
                bonus = self.c * np.sqrt(2.0 * np.log(self.round + 1) / len(self.rewards[a]))
                scores[a] = mean + bonus

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        reward = float(reward)
        self.rewards[arm].append(reward)

        # Update best arm tracking based on window mean
        if len(self.rewards[arm]) >= 5:
            current_mean = np.mean(self.rewards[arm])
            if current_mean > self.best_reward:
                self.best_reward = current_mean
                self.best_arm = arm
# EVOLVE-BLOCK-END
