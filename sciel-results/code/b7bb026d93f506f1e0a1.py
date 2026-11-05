# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global linear bandit with one-hot arm indicators, adaptive forgetting,
    and deadline guard.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: context + one-hot arm indicators
        self.d_aug = dim + n_arms

        # Global model
        self.A = np.eye(self.d_aug)
        self.b = np.zeros(self.d_aug)
        self.A_inv = np.eye(self.d_aug)

        # Forgetting schedule
        self.gamma = 0.95
        self.gamma_min = 0.75
        self.gamma_decay = 0.9999
        self.ridge = 0.15
        self.C = 1.0
        self.C_decay = 0.99995

        # Deadline guard
        self.t0 = time.time()
        self.deadline = self.t0 + 135.0  # safety margin before 150s kill

        # Track step count for adaptive exploration
        self.step = 0

    def _augment(self, context, arm):
        """Create augmented feature vector: [context, one-hot arm]."""
        x = np.zeros(self.d_aug)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline check: if we're running out of time, be greedy
        if time.time() > self.deadline:
            # Greedy fallback
            scores = []
            for a in range(self.n):
                x = self._augment(context, a)
                theta = self.A_inv @ self.b
                scores.append(theta @ x)
            return int(np.argmax(scores))

        scores = []
        theta = self.A_inv @ self.b
        for a in range(self.n):
            x = self._augment(context, a)
            mu = theta @ x
            # Exploration bonus: sqrt(x^T A_inv x) * C
            bonus = self.C * np.sqrt(max(0.0, x @ self.A_inv @ x))
            scores.append(mu + bonus)

        arm = int(np.argmax(scores))
        self.step += 1

        # Adaptive forgetting: decrease gamma over time to adapt faster
        # but floor at gamma_min
        self.gamma = max(self.gamma_min, self.gamma * self.gamma_decay)
        # Decrease C over time to reduce exploration
        self.C = max(0.5, self.C * self.C_decay)

        return arm

    def update(self, context, arm, reward):
        x = self._augment(context, arm)

        # Update with forgetting
        self.A = self.gamma * self.A + np.outer(x, x)
        self.b = self.gamma * self.b + reward * x

        # Add ridge for stability
        self.A += self.ridge * np.eye(self.d_aug)

        # Recompute inverse (for small d_aug this is fine)
        # Use Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(self.A)
            self.A_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Fallback to direct inverse
            self.A_inv = np.linalg.inv(self.A + 1e-6 * np.eye(self.d_aug))
# EVOLVE-BLOCK-END
