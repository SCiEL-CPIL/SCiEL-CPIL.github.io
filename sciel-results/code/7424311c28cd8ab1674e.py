"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with per-arm adaptive forgetting and Cholesky-based variance for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta_cache = [None for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        self.base_forgetting = 0.995
        self.forgetting_rates = [self.base_forgetting for _ in range(n_arms)]
        
        # EWMA of squared residuals for non-stationarity detection (faster adaptation)
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.residual_ewma_alpha = 0.1  # Faster adaptation
        
        # Exploration parameter
        self.exploration_alpha = 0.7
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 120.0  # Safety margin before 150s kill

    def _update_theta_cache(self, arm):
        """Update cached theta for an arm."""
        try:
            self.theta_cache[arm] = np.linalg.solve(self.A[arm] + 1e-6 * np.eye(self.d), self.b[arm])
        except np.linalg.LinAlgError:
            self.theta_cache[arm] = np.linalg.pinv(self.A[arm] + 1e-6 * np.eye(self.d)) @ self.b[arm]

    def _get_forgetting(self, arm, context, reward):
        """Adapt forgetting rate based on observed non-stationarity."""
        if self.t > 0:
            # Use cached theta if available
            if self.theta_cache[arm] is not None:
                theta = self.theta_cache[arm]
            else:
                self._update_theta_cache(arm)
                theta = self.theta_cache[arm]
            
            pred = theta @ context
            residual = reward - pred
            residual_sq = residual ** 2
            
            # Update EWMA of squared residuals
            self.residual_sq_ewma[arm] = (1 - self.residual_ewma_alpha) * self.residual_sq_ewma[arm] + self.residual_ewma_alpha * residual_sq
            
            # If residual is large, increase forgetting (lower alpha)
            # Base forgetting is 0.995, max forgetting is 0.85 (more aggressive)
            # Scale based on residual magnitude
            if self.residual_sq_ewma[arm] > 0.05:
                # More aggressive forgetting when non-stationarity detected
                self.forgetting_rates[arm] = max(0.85, self.base_forgetting - 0.15 * np.sqrt(self.residual_sq_ewma[arm]))
            else:
                # Gradually return to base forgetting
                self.forgetting_rates[arm] += 0.002 * (self.base_forgetting - self.forgetting_rates[arm])
        
        return self.forgetting_rates[arm]

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return the arm with highest mean reward so far
            means = []
            for a in range(self.n):
                if self.theta_cache[a] is not None:
                    means.append(self.theta_cache[a] @ context)
                else:
                    self._update_theta_cache(a)
                    means.append(self.theta_cache[a] @ context)
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            # Use Cholesky for stable variance computation
            try:
                A_reg = self.A[a] + 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(A_reg)
                # Compute theta using solve
                theta = np.linalg.solve(A_reg, self.b[a])
                self.theta_cache[a] = theta
                mean = theta @ context
                # Variance: context^T A^{-1} context = ||A^{-1/2} context||^2
                v = np.linalg.solve(L, context)
                variance = np.dot(v, v)
                scores.append(mean + self.exploration_alpha * np.sqrt(variance))
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                A_reg = self.A[a] + 1e-6 * np.eye(self.d)
                theta = np.linalg.pinv(A_reg) @ self.b[a]
                self.theta_cache[a] = theta
                mean = theta @ context
                variance = context @ np.linalg.pinv(A_reg) @ context
                scores.append(mean + self.exploration_alpha * np.sqrt(variance))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Get adaptive forgetting rate
        alpha = self._get_forgetting(arm, context, reward)
        
        # Update with forgetting
        self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = alpha * self.b[arm] + reward * context
        
        # Invalidate theta cache for this arm
        self.theta_cache[arm] = None
        
        self.t += 1
# EVOLVE-BLOCK-END
