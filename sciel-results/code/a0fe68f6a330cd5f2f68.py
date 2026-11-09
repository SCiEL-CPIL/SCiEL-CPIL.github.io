"""
Adaptive Sliding Window LinUCB for Non-Stationary Linear Bandits.
Uses an adaptive window size based on recent prediction errors to balance
reactivity to changes and estimation stability.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        
        # Parameters
        self.reg = 1.0
        self.beta = 1.2  # Exploration constant
        self.base_gamma = 0.95  # Base forgetting factor
        self.min_gamma = 0.85   # Most reactive (fast forgetting)
        self.max_gamma = 0.99   # Most stable (slow forgetting)
        
        # Adaptive forgetting state
        self.gammas = [self.base_gamma] * n_arms
        self.residuals = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Per-arm data structures
        # We store weighted (context, reward) pairs
        self.histories = [deque() for _ in range(n_arms)]
        
        # Pre-computed matrices
        self.A_inv = [None] * n_arms
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin

    def _recompute_arm(self, arm):
        """Recompute A, b, and A_inv for a specific arm using exponential weighting."""
        if len(self.histories[arm]) == 0:
            self.A[arm] = self.reg * np.eye(self.dim)
            self.b[arm] = np.zeros(self.dim)
            self.A_inv[arm] = np.eye(self.dim) / self.reg
            return

        # Build A and b with exponential weights (most recent = highest weight)
        A = self.reg * np.eye(self.dim)
        b = np.zeros(self.dim)
        
        gamma = self.gammas[arm]
        n = len(self.histories[arm])
        
        # Apply exponential weighting: weight_i = gamma^(n-1-i)
        for i, (ctx, rew) in enumerate(self.histories[arm]):
            weight = gamma ** (n - 1 - i)
            A += weight * np.outer(ctx, ctx)
            b += weight * rew * ctx
            
        self.A[arm] = A
        self.b[arm] = b
        
        # Use Cholesky for better numerical stability
        try:
            L = np.linalg.cholesky(A)
            self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            try:
                L = np.linalg.cholesky(A + 1e-6 * np.eye(self.dim))
                self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.linalg.inv(A + 1e-4 * np.eye(self.dim))

    def _adapt_forgetting(self, arm, residual):
        """Adapt the forgetting factor for the arm based on recent residuals."""
        self.residuals[arm].append(abs(residual))
        
        if len(self.residuals[arm]) < 5:
            return
        
        mean_res = np.mean(self.residuals[arm])
        current_gamma = self.gammas[arm]
        
        # If residuals are high, the model is likely stale -> increase forgetting (lower gamma)
        # If residuals are low, the model is good -> decrease forgetting (higher gamma)
        
        if mean_res > 0.8:
            # High error: forget faster
            new_gamma = max(self.min_gamma, current_gamma - 0.02)
        elif mean_res < 0.05:
            # Low error: forget slower for stability
            new_gamma = min(self.max_gamma, current_gamma + 0.01)
        else:
            # Moderate: slowly return to base
            new_gamma = current_gamma + (self.base_gamma - current_gamma) * 0.1
            
        if abs(new_gamma - current_gamma) > 1e-6:
            self.gammas[arm] = new_gamma
            # Full recompute needed when gamma changes
            self._recompute_arm(arm)

    def select(self, context):
        if time.time() - self.start_time > self.deadline:
            return 0

        context = np.asarray(context, dtype=np.float64)
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Ensure A_inv is up to date
            if self.A_inv[a] is None:
                self._recompute_arm(a)
            
            # LinUCB score
            theta = self.A_inv[a] @ self.b[a]
            mu = theta @ context
            variance = context @ self.A_inv[a] @ context
            score = mu + self.beta * np.sqrt(variance)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        if time.time() - self.start_time > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        
        # Calculate residual for adaptation
        if self.A_inv[arm] is not None and len(self.histories[arm]) > 0:
            theta = self.A_inv[arm] @ self.b[arm]
            pred = theta @ context
            residual = reward - pred
        else:
            residual = 0.0
            
        # Adapt forgetting factor
        self._adapt_forgetting(arm, residual)
        
        # Add new observation
        self.histories[arm].append((context, reward))
        
        # Cap history length to avoid unbounded memory (effective horizon ~ log(1/eps)/-log(gamma))
        max_hist = 200
        if len(self.histories[arm]) > max_hist:
            self.histories[arm].popleft()
            self._recompute_arm(arm)
        else:
            # Incremental update with exponential forgetting:
            # A_new = gamma * A_old + x x^T
            # b_new = gamma * b_old + reward * x
            gamma = self.gammas[arm]
            
            # First, decay existing matrices
            self.A[arm] *= gamma
            self.b[arm] *= gamma
            
            # Then add new observation
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context
            
            # Update inverse using Sherman-Morrison-Woodbury on the scaled matrix
            # We need to be careful: A_inv was for the old A, now we have gamma*A_old + xx^T
            # Simplest: recompute inverse from scratch (dim is small)
            try:
                L = np.linalg.cholesky(self.A[arm])
                self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                try:
                    L = np.linalg.cholesky(self.A[arm] + 1e-6 * np.eye(self.dim))
                    self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    self.A_inv[arm] = np.linalg.inv(self.A[arm] + 1e-4 * np.eye(self.dim))
        
        self.t += 1
# EVOLVE-BLOCK-END
