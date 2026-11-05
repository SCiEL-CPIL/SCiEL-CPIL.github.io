# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

class Policy:
    """Sliding Window Linear Bandit (SW-UCB)."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 100  # Sliding window size
        self.lambda_reg = 1.0   # Regularization parameter
        self.beta = 1.5         # Exploration parameter
        
        # Initialize A (design matrix) and b (reward vector) for each arm
        # A includes the regularization term lambda * I
        self.A = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        
        # Store history for each arm to allow removing old data points
        self.history = [deque(maxlen=self.window_size) for _ in range(self.n)]
        
        self.t = 0

    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            try:
                # Use solve for stability, A is symmetric positive definite
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                theta = np.linalg.pinv(self.A[a]) @ self.b[a]
            
            pred = theta @ context
            
            # Compute variance: context^T A^{-1} context
            # We can compute A_inv once if we need it for multiple contexts, 
            # but here we have one context per select call.
            # To avoid explicit inverse, solve A * v = context, then v = A^-1 context
            # var = context @ v
            try:
                v = np.linalg.solve(self.A[a], context)
                var = context @ v
            except np.linalg.LinAlgError:
                v = np.linalg.pinv(self.A[a]) @ context
                var = context @ v
            
            # UCB score
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # If the window is full, remove the oldest data point
        if len(self.history[arm]) == self.window_size:
            old_context, old_reward = self.history[arm][0]
            self.A[arm] -= np.outer(old_context, old_context)
            self.b[arm] -= old_reward * old_context
            self.history[arm].popleft()
        
        # Add the new data point
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.history[arm].append((context, reward))
        
        self.t += 1
# EVOLVE-BLOCK-END
