"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with forgetting factor and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.beta = 0.99
        self.alpha = 1.0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 5s safety margin

    def select(self, context):
        # Deadline guard: return a safe default if time is running out
        if time.time() > self.deadline:
            return 0
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                variance = context @ inv @ context
                scores.append(theta @ context + self.alpha * np.sqrt(max(variance, 0)))
            except np.linalg.LinAlgError:
                scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
