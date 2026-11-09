"""
Adaptive Sliding Window LinUCB for Non-Stationary Linear Bandits.

Uses a sliding window of recent observations for each arm to maintain a regularized
linear regression model. This approach allows the policy to completely forget
old data after a fixed window, providing faster adaptation to non-stationarity
compared to exponential forgetting. Uses Cholesky decomposition for numerical
stability and efficient computation of predictions and confidence bounds.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Sliding window parameters
        self.window_size = 30  # Number of recent samples to keep per arm
        self.reg = 1.0         # Regularization parameter (ridge)
        self.alpha = 1.5       # Exploration parameter for UCB
        
        # Data storage for sliding window
        self.contexts = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.rewards = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Linear regression state: A = reg*I + sum(x x^T), b = sum(r x)
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached Cholesky factor and weights for fast select
        # A = L @ L.T
        self.L = [np.linalg.cholesky(self.A[i]) for i in range(n_arms)]
        self.theta = [np.linalg.solve(self.L[i].T, np.linalg.solve(self.L[i], self.b[i])) for i in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Predicted mean
            pred = context @ self.theta[a]
            
            # Confidence term: sqrt(x^T A_inv x) = || L^-1 x ||_2
            # Solve L z = x
            z = np.linalg.solve(self.L[a], context)
            conf = np.linalg.norm(z)
            
            # UCB score
            scores[a] = pred + self.alpha * conf
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # If window is full, remove the oldest sample
        if len(self.contexts[arm]) == self.window_size:
            old_ctx = self.contexts[arm][0]
            old_rew = self.rewards[arm][0]
            
            # Incrementally subtract from A and b
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_rew * old_ctx
            
        # Add new sample
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
        
        # Incrementally add to A and b
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Recompute Cholesky factor and weights for the updated arm only
        # Ensure numerical stability
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            # Solve L y = b, then L.T theta = y
            y = np.linalg.solve(self.L[arm], self.b[arm])
            self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
        except np.linalg.LinAlgError:
            # If singular or not positive definite, add small regularization
            self.A[arm] += 1e-6 * np.eye(self.dim)
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                y = np.linalg.solve(self.L[arm], self.b[arm])
                self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky still fails
                self.L[arm] = np.linalg.cholesky(self.A[arm] + 1e-4 * np.eye(self.dim))
                y = np.linalg.solve(self.L[arm], self.b[arm])
                self.theta[arm] = np.linalg.solve(self.L[arm].T, y)
# EVOLVE-BLOCK-END
