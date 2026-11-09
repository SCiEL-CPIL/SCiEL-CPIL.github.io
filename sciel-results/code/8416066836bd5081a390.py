"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit policy using a multi-timescale approach.
    Maintains a short-window (fast-adapting) and long-window (stable) estimator.
    Uses Thompson Sampling with the short-window posterior for exploration,
    regularized by the long-window mean for stability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Regularization parameter

        # Short window: fast forgetting for adaptation
        self.gamma_short = 0.95
        # Long window: slow forgetting for stability
        self.gamma_long = 0.995

        # Short-window estimators
        self.A_short = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]

        # Long-window estimators
        self.A_long = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]

        # UCB/TS parameters
        self.alpha = 1.0  # Exploration parameter

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Precompute inverse Cholesky or use direct inverse
        # For efficiency, we'll use np.linalg.solve where possible

    def _get_theta_cov(self, A, b):
        """Compute theta and covariance from A and b."""
        # theta = A^{-1} b
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
        # Covariance = A^{-1}
        # For efficiency in TS, we can use the Cholesky factor
        try:
            L = np.linalg.cholesky(A)
            # Cov = A^{-1} = L^{-T} L^{-1}
            # We'll compute the inverse for sampling
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            cov = np.eye(self.d)
        return theta, cov

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Return a default arm (e.g., the one with highest long-window mean)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta_long, _ = self._get_theta_cov(self.A_long[a], self.b_long[a])
                score = theta_long @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Thompson Sampling with short-window posterior
        # Regularize with long-window mean
        sampled_values = []
        for a in range(self.n):
            # Get short-window posterior
            theta_short, cov_short = self._get_theta_cov(self.A_short[a], self.b_short[a])
            # Get long-window mean for regularization
            theta_long, _ = self._get_theta_cov(self.A_long[a], self.b_long[a])

            # Blend: use short-window for exploration, long-window for stability
            # Sample from short-window posterior, but pull toward long-window mean
            # This is a heuristic: sample from N(theta_short, cov_short)
            # but if the long-window mean is very different, trust long-window more
            try:
                # Sample from the short-window posterior
                theta_sample = np.random.multivariate_normal(theta_short, cov_short)
            except np.linalg.LinAlgError:
                theta_sample = theta_short

            # Compute value
            value = theta_sample @ context

            # Add a small bonus based on disagreement between short and long
            # If they disagree, it might be a shift, so explore more
            disagreement = np.linalg.norm(theta_short - theta_long)
            bonus = 0.1 * disagreement * np.sqrt(context @ cov_short @ context)

            sampled_values.append(value + bonus)

        return int(np.argmax(sampled_values))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        outer = np.outer(context, context)

        # Update short-window
        self.A_short[arm] = self.gamma_short * self.A_short[arm] + outer
        self.b_short[arm] = self.gamma_short * self.b_short[arm] + reward * context

        # Update long-window
        self.A_long[arm] = self.gamma_long * self.A_long[arm] + outer
        self.b_long[arm] = self.gamma_long * self.b_long[arm] + reward * context
# EVOLVE-BLOCK-END
