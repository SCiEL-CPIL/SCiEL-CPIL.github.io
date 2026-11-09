"""Adaptive Exponential Forgetting Thompson Sampling for Non-Stationary Linear Bandits."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Uses Discounted Least Squares (DLS) with an adaptive forgetting factor.
    
    Structural change from Sliding Window to Recursive Exponential Forgetting:
    - Maintains running statistics A and b, decayed by gamma at each step.
    - Gamma is adapted based on prediction error (residuals).
    - Faster adaptation to change when residuals are high, more stability when low.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Regularization
        self.reg = 1.0
        
        # Thompson Sampling noise variance
        self.sigma_sq = 1.0
        
        # Per-arm statistics: A (covariance inverse), b (moment)
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        # Base forgetting factor (closer to 1 means longer memory)
        self.base_gamma = 0.98
        self.min_gamma = 0.85   # Fastest forgetting (shortest memory)
        self.max_gamma = 0.995  # Slowest forgetting (longest memory)
        
        # EWMA of squared residuals to drive adaptation
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.2
        
        # Current gamma for each arm
        self.gamma = [self.base_gamma for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def _update_gamma(self, arm, residual_sq):
        """
        Update the forgetting factor based on the magnitude of the residual.
        Large residual -> model is wrong -> forget old data faster (lower gamma).
        Small residual -> model is good -> keep memory (higher gamma).
        """
        # Normalize residual to [0, 1] roughly. 
        # Assuming rewards are O(1). If residual > 1.0, we are in "high error" zone.
        # Map residual_sq to a weight.
        # If residual_sq is large, we want gamma close to min_gamma.
        # If residual_sq is small, we want gamma close to max_gamma.
        
        # Simple mapping:
        # factor = 1 / (1 + residual_sq * scale)
        # gamma = min_gamma + (max_gamma - min_gamma) * factor
        
        # Let's use a sigmoid-like or inverse mapping.
        # Scale factor to determine sensitivity.
        scale = 1.0 
        factor = 1.0 / (1.0 + self.residual_sq_ewma[arm] * scale)
        
        new_gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * factor
        
        # Smooth the gamma update to avoid oscillation
        self.gamma[arm] = 0.9 * self.gamma[arm] + 0.1 * new_gamma

    def _recompute_arm_stats(self, arm):
        """Recompute inverse and theta from current A and b."""
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            # Fallback to regularized identity if singular
            self.A_inv[arm] = np.eye(self.d) / self.reg
            self.theta[arm] = np.zeros(self.d)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            context = np.asarray(context, dtype=float)
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            theta_mean = self.theta[a]
            
            # Sample theta from posterior N(theta_mean, sigma^2 * A_inv[a])
            try:
                L = np.linalg.cholesky(self.A[a])
                z = np.random.randn(self.d)
                w = np.linalg.solve(L.T, z)
                theta_sample = theta_mean + self.sigma_sq * w
            except np.linalg.LinAlgError:
                theta_sample = theta_mean
            
            score = theta_sample @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        reward = float(reward)
        
        # 1. Compute residual
        pred = self.theta[arm] @ context
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update EWMA of squared residuals
        self.residual_sq_ewma[arm] = (1 - self.ewma_alpha) * self.residual_sq_ewma[arm] + self.ewma_alpha * residual_sq
        
        # 3. Update adaptive gamma
        self._update_gamma(arm, residual_sq)
        
        # 4. Update statistics with forgetting
        # Decay old information
        gamma = self.gamma[arm]
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # 5. Recompute inverse and theta
        self._recompute_arm_stats(arm)
        
        self.t += 1
# EVOLVE-BLOCK-END
