"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices (A) and mean vectors (b)
        # A is the precision matrix (inverse covariance), b is the weighted sum of contexts
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor: 0.99 provides a balance between memory and adaptability
        self.alpha = 0.99
        # Assumed noise variance for Thompson Sampling
        self.sigma_sq = 1.0
        # Initial exploration bonus
        self.lambda0 = 1.0
        
        # Track elapsed time for safety
        self._start_time = time.time()
        self._deadline = 120.0  # seconds

    def _compute_posterior(self, arm):
        """Compute the posterior mean and covariance for a given arm."""
        inv_A = np.linalg.inv(self.A[arm])
        theta = inv_A @ self.b[arm]
        return theta, inv_A

    def select(self, context):
        # Check deadline
        if time.time() - self._start_time > self._deadline - 2.0:
            # Return a random arm as a safety fallback
            return int(np.random.randint(self.n))
            
        scores = []
        for a in range(self.n):
            theta, inv_A = self._compute_posterior(a)
            
            # Sample from the posterior
            # theta ~ N(theta_hat, inv_A)
            sampled_theta = theta + np.random.multivariate_normal(np.zeros(self.d), inv_A)
            
            # Score is the predicted reward
            scores.append(sampled_theta @ context)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() - self._start_time > self._deadline - 2.0:
            return
            
        # Update with forgetting factor
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
        # Regularization to prevent singularity
        self.A[arm] += 1e-6 * np.eye(self.d)
# EVOLVE-BLOCK-END
