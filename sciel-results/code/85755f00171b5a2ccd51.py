"""Non-stationary linear contextual bandit with Discounted Thompson Sampling."""
import numpy as np


class Policy:
    """Discounted Linear Thompson Sampling for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.97  # Forgetting factor (faster than seed's 0.995)
        self.lam = 1.0     # Regularization parameter
        self.t = 0         # Time step counter

        # Initialize per-arm state
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Pre-allocate for speed
        self._inv_cache = [None] * n_arms
        self._theta_cache = [None] * n_arms

    def select(self, context):
        """Select an arm using Thompson Sampling with discounted estimates."""
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            # Compute inverse of A (with numerical stabilization)
            try:
                inv_A = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                inv_A = np.linalg.pinv(self.A[a])

            # Posterior mean
            theta = inv_A @ self.b[a]

            # Posterior covariance for the context
            cov = context @ inv_A @ context
            std = np.sqrt(max(cov, 0.0))

            # Sample from posterior
            scores[a] = theta @ context + std * np.random.randn()

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the per-arm statistics with a forgetting factor."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Discount previous observations
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Invalidate cache for this arm
        self._inv_cache[arm] = None
        self._theta_cache[arm] = None
