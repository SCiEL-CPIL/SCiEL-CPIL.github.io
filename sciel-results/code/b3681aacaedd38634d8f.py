"""Non-stationary linear contextual bandit with adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


class Policy:
    """LinUCB with adaptive forgetting factor for non-stationary rewards."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 0.1  # Regularization
        self.gamma = 0.995  # Forgetting factor
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) / self.lam for _ in range(n_arms)]
        self.alpha = 1.0  # Exploration parameter
        self.reward_history = [deque(maxlen=100) for _ in range(n_arms)]
        self.arm_counts = [0 for _ in range(n_arms)]

    def select(self, context):
        scores = []
        for a in range(self.n):
            theta_a = self.theta[a]
            mean = theta_a @ context
            # Uncertainty term
            var = context @ self.inv_A[a] @ context
            scores.append(mean + self.alpha * np.sqrt(max(var, 0)))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.arm_counts[arm] += 1
        self.reward_history[arm].append(reward)

        # Update with forgetting factor
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Compute inverse
        self.inv_A[arm] = np.linalg.inv(self.A[arm])
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

        # Adapt forgetting factor based on reward change
        if len(self.reward_history[arm]) >= 10:
            recent = list(self.reward_history[arm])
            if len(recent) >= 20:
                diff = np.diff(recent[-20:])
                variance = np.var(diff)
                # Higher variance -> lower gamma (faster forgetting)
                target_gamma = 0.995 - 0.03 * min(variance / (1 + variance), 1.0)
                self.gamma = 0.9 * self.gamma + 0.1 * target_gamma
                self.gamma = float(np.clip(self.gamma, 0.95, 0.999))
# EVOLVE-BLOCK-END
