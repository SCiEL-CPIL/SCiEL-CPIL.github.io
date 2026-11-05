"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary Linear Bandit with adaptive forgetting and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]  # Regularization
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.gamma = 0.98  # Forgetting factor
        self.alpha = 1.0   # Exploration parameter
        self.round = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110  # Safety margin before 120s limit

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a default arm if we're running out of time
            return 0

        self.round += 1
        scores = []
        for a in range(self.n):
            try:
                # Compute theta estimate
                theta = np.linalg.solve(self.A[a], self.b[a])
                # Compute uncertainty
                A_inv = np.linalg.inv(self.A[a])
                uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
                scores.append(theta @ context + uncertainty)
            except np.linalg.LinAlgError:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Adaptive forgetting: adjust gamma based on recent performance
        # Use a simple adaptive scheme
        self.gamma = 0.98 + 0.02 * np.exp(-self.round / 1000)

        # Update with forgetting
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Periodically re-regularize to prevent ill-conditioning
        if self.round % 100 == 0:
            self.A[arm] += 0.1 * np.eye(self.d)
# EVOLVE-BLOCK-END
