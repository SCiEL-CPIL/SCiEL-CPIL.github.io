"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with adaptive forgetting for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.forgetting = 0.99
        self.noise_var = 1.0
        self.t = 0
        self.recent_rewards = [[] for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 110.0  # Safety margin before 120s limit

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard
        if time.time() > self._deadline:
            # Return best arm based on current point estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            # Sample from posterior: theta_a ~ N(theta, (noise_var * A_inv))
            # For efficiency, use Cholesky or eigen decomposition
            try:
                # Covariance is noise_var * A_inv
                cov = self.noise_var * self.A_inv[a]
                # Use eigen decomposition for sampling (more stable for large matrices)
                # Actually, for small dims, direct method is fine
                # Sample: theta + L @ z where cov = L @ L.T
                # Use np.random.multivariate_normal for correctness, but it's slow
                # For dim <= 10, let's use a faster method
                # Cholesky decomposition
                L = np.linalg.cholesky(cov)
                z = np.random.randn(self.d)
                theta_sample = theta + L @ z
            except np.linalg.LinAlgError:
                # Fallback to point estimate if Cholesky fails
                theta_sample = theta
            
            scores.append(theta_sample @ context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Track recent rewards for adaptive forgetting
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > 20:
            self.recent_rewards[arm].pop(0)
        
        # Adaptive forgetting factor based on reward variance in recent window
        if len(self.recent_rewards[arm]) >= 5:
            recent = self.recent_rewards[arm]
            variance = np.var(recent)
            # Higher variance -> lower forgetting (more memory)
            # Lower variance -> higher forgetting (adapt faster)
            # Map variance to forgetting factor in [0.95, 0.999]
            self.forgetting = 0.95 + 0.049 * min(1.0, variance / 2.0)
        
        # Update with forgetting
        gamma = self.forgetting
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Update inverse using Sherman-Morrison or direct inversion
        # For small dims, direct inversion is acceptable
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Fallback: add small regularization
            self.A[arm] += 1e-6 * np.eye(self.d)
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
        
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]
# EVOLVE-BLOCK-END
