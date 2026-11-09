"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary Linear Contextual Bandit Policy with adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.alpha = 0.995
        self.recent_errors = []
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        self._precompute_inv()

    def _precompute_inv(self):
        self.inv_A = [np.linalg.inv(self.A[a]) for a in range(self.n)]
        self.theta = [self.inv_A[a] @ self.b[a] for a in range(self.n)]

    def select(self, context):
        if time.time() > self.deadline:
            return 0

        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            pred = theta @ context
            # Compute variance term using precomputed inverse
            var = context @ self.inv_A[a] @ context
            ucb = pred + 0.5 * np.sqrt(var)
            scores.append(ucb)

        best_arm = int(np.argmax(scores))
        self._last_arm = best_arm
        return best_arm

    def update(self, context, arm, reward):
        # Update alpha based on recent prediction error
        if hasattr(self, '_last_arm') and arm == self._last_arm:
            pred = self.theta[arm] @ context
            error = abs(reward - pred)
            self.recent_errors.append(error)
            if len(self.recent_errors) > 10:
                self.recent_errors.pop(0)

            avg_error = np.mean(self.recent_errors)
            # If error is high, increase forgetting (lower alpha)
            # If error is low, decrease forgetting (higher alpha)
            # Map avg_error to alpha range [0.98, 0.999]
            if avg_error > 1.0:
                self.alpha = 0.98
            elif avg_error > 0.5:
                self.alpha = 0.99
            elif avg_error > 0.2:
                self.alpha = 0.995
            else:
                self.alpha = 0.999

        # Update A and b with forgetting
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context

        # Recompute inverse and theta for the updated arm
        self.inv_A[arm] = np.linalg.inv(self.A[arm])
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]
# EVOLVE-BLOCK-END
