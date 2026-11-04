"""Non-stationary linear contextual bandit with EMA weights and Epsilon-Greedy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Exponential Moving Average (EMA)
    of feature-reward products and Epsilon-Greedy selection.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # EMA weights for each arm: theta[a] approximates E[r*x | arm=a]
        # Initialized to zero
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Learning rate for EMA (decay factor)
        # Higher alpha = faster adaptation to new data, but noisier estimates
        # Lower alpha = smoother, but slower to adapt to non-stationarity
        self.alpha = 0.1

        # Epsilon for exploration
        self.epsilon = 0.1

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

    def _check_deadline(self):
        """Return True if we should stop computing and use a safe fallback."""
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def select(self, context):
        """Select arm using Epsilon-Greedy based on EMA scores."""
        if self._check_deadline():
            # Fallback: pick arm with highest mean estimate
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        # Epsilon-greedy: explore with probability epsilon
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n))

        # Exploit: pick arm with highest estimated reward
        scores = np.array([self.theta[a] @ context for a in range(self.n)])
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update EMA weights for the selected arm."""
        context = np.asarray(context, dtype=np.float64)

        # Update theta for the chosen arm
        # theta = (1 - alpha) * theta + alpha * (reward * context)
        self.theta[arm] = (1.0 - self.alpha) * self.theta[arm] + self.alpha * (reward * context)
# EVOLVE-BLOCK-END
