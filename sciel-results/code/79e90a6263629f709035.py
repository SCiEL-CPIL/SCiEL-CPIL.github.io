"""Non-stationary linear contextual bandit with global linear model and Thompson Sampling."""
import numpy as np
import time


class Policy:
    """
    Global linear model with context-augmented features (one-hot arm + context).
    Uses Thompson Sampling for exploration and a forgetting factor for non-stationarity.
    Includes a deadline guard to ensure the program finishes within the time limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: one-hot arm (n) + context (d)
        self.aug_dim = n_arms + dim
        # Global model parameters
        self.A = np.eye(self.aug_dim)
        self.b = np.zeros(self.aug_dim)
        # Forgetting factor for non-stationarity
        self.alpha = 0.98
        # Noise variance for Thompson Sampling
        self.sigma_sq = 1.0
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110  # 110s safety margin before 120s limit
        self.round_count = 0

    def _augment_context(self, context, arm):
        """Create augmented feature vector: one-hot arm encoding + context."""
        aug = np.zeros(self.aug_dim)
        aug[arm] = 1.0
        aug[self.n:] = context
        return aug

    def select(self, context):
        self.round_count += 1
        # Deadline guard: if we're getting close to the limit, return a fixed arm
        if time.time() > self.deadline:
            return 0

        # Compute posterior mean and covariance for Thompson Sampling
        # Use Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(self.A)
            # Solve L L^T theta = b
            theta_hat = np.linalg.solve(L.T, np.linalg.solve(L, self.b))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            theta_hat = np.linalg.pinv(self.A) @ self.b

        # Compute posterior covariance: (A^{-1} - A^{-1} b b^T A^{-1} / sigma_sq)
        # Actually for linear regression with known noise, the posterior covariance is A^{-1}
        # and the posterior mean is A^{-1} b
        # For Thompson Sampling, we sample from N(theta_hat, A^{-1})
        
        # Compute A^{-1} via Cholesky
        try:
            L = np.linalg.cholesky(self.A)
            # A^{-1} = L^{-T} L^{-1}
            # We need to sample from N(theta_hat, A^{-1})
            # Sample z ~ N(0, I), then theta = theta_hat + L^{-T} z
            z = np.random.randn(self.aug_dim)
            # Solve L w = z
            w = np.linalg.solve(L, z)
            theta_sample = theta_hat + w
        except np.linalg.LinAlgError:
            theta_sample = theta_hat

        # Compute scores for each arm
        scores = []
        for a in range(self.n):
            aug = self._augment_context(context, a)
            scores.append(theta_sample @ aug)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        # Deadline guard
        if time.time() > self.deadline:
            return

        aug = self._augment_context(context, arm)
        
        # Update with forgetting factor
        self.A = self.alpha * self.A + np.outer(aug, aug)
        self.b = self.alpha * self.b + reward * aug
