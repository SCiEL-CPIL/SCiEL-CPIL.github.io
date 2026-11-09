"""Non-stationary linear contextual bandit with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Discounted LinUCB with adaptive exploration and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.97          # Forgetting factor
        self.lam = 1.0             # Ridge regularization
        self.beta0 = 1.5           # Base exploration constant
        self.deadline = time.time() + 110.0  # Safety margin before 120s limit

        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.reward_history = [[] for _ in range(n_arms)]
        self.round_count = 0
        self._inv_cache = [None] * n_arms

    def _get_inv(self, a):
        """Compute and cache inverse of A[a], recomputing if needed."""
        if self._inv_cache[a] is None:
            try:
                self._inv_cache[a] = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                self._inv_cache[a] = np.eye(self.d)
        return self._inv_cache[a]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64).ravel()
        scores = np.zeros(self.n)

        for a in range(self.n):
            inv = self._get_inv(a)
            theta = inv @ self.b[a]
            mean = theta @ context

            # Adaptive exploration: scale beta by recent reward variance
            beta = self.beta0
            hist = self.reward_history[a]
            if len(hist) >= 5:
                recent = hist[-20:]
                var = np.var(recent)
                beta = self.beta0 * (1.0 + 0.5 * np.sqrt(var + 1e-8))

            # UCB bonus
            var_term = context @ inv @ context
            bonus = beta * np.sqrt(max(var_term, 0.0))
            scores[a] = mean + bonus

        self.round_count += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).ravel()

        # Discounted update
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Invalidate inverse cache for this arm
        self._inv_cache[arm] = None

        # Track reward history (bounded)
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > 100:
            self.reward_history[arm].pop(0)

        # Periodically re-regularize to prevent A from becoming ill-conditioned
        if self.round_count % 50 == 0:
            for a in range(self.n):
                self.A[a] = self.lam * np.eye(self.d) + self.A[a]
                self._inv_cache[a] = None
# EVOLVE-BLOCK-END
