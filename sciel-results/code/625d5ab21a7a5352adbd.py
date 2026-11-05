"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Discounted Thompson Sampling for Non-Stationary Linear Bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        
        # Regularization for stability
        self.reg = 1.0
        
        # Initialize sufficient statistics
        # A[a] is the precision matrix (inverse covariance)
        # b[a] is the sufficient statistic for the mean
        self.A = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        self.alpha_min = 0.9
        self.alpha_max = 0.999
        self.alpha = 0.99  # Initial forgetting factor
        
        # Error tracking for adaptive alpha
        self.err_avg = 1.0  # Initialize with a reasonable guess
        self.err_delta = 0.1  # Learning rate for error average
        
        # Pre-compute Cholesky factors? No, they change every update.
        # We will compute them on the fly in select/update.

    def _get_deadline_status(self):
        """Check if we are close to the time limit."""
        elapsed = time.time() - self.start_time
        return elapsed > 115.0  # Safety margin

    def select(self, context):
        if self._get_deadline_status():
            # Fallback: Greedy based on current means
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        # Thompson Sampling
        scores = []
        for a in range(self.n):
            try:
                # Compute mean and covariance
                theta_mean = np.linalg.solve(self.A[a], self.b[a])
                cov = np.linalg.inv(self.A[a])
                
                # Sample from posterior
                # Use Cholesky for sampling
                L = np.linalg.cholesky(cov + 1e-9 * np.eye(self.d)) # small jitter for stability
                z = np.random.randn(self.d)
                theta_sample = theta_mean + L @ z
                
                scores.append(theta_sample @ context)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._get_deadline_status():
            return

        # Compute current mean for residual calculation
        try:
            theta_mean = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta_mean @ context
            residual = reward - pred
        except np.linalg.LinAlgError:
            residual = reward  # Fallback if solve fails

        # Update error average
        self.err_avg = (1 - self.err_delta) * self.err_avg + self.err_delta * (residual ** 2)
        
        # Adapt forgetting factor alpha
        # If error is high, alpha should be low (fast forgetting)
        # If error is low, alpha should be high (slow forgetting)
        # alpha = alpha_min + (alpha_max - alpha_min) * exp(-k * err_avg)
        k = 5.0  # Sensitivity parameter
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-k * self.err_avg)
        
        # Update sufficient statistics with adaptive forgetting
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
