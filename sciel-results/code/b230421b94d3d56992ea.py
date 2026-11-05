"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with a forgetting factor and adaptive exploration."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.alpha = 0.5
        self.gamma = 0.995
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.reward_sums = [0.0 for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        self.round = 0
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

    def _is_deadline(self):
        return (time.time() - self.start_time) >= self.deadline

    def select(self, context):
        self.round += 1
        context = np.asarray(context, dtype=np.float64)

        # If we're close to deadline, just return argmax of current estimates
        if self._is_deadline():
            scores = []
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mean = theta @ context
                # Adaptive alpha: reduce exploration as we gather more data
                # but keep it responsive to non-stationarity
                local_alpha = self.alpha * (1.0 + 0.5 / (1.0 + self.counts[a]))
                var = context @ inv @ context
                ucb = mean + local_alpha * np.sqrt(max(var, 0.0))
                scores.append(ucb)
            except np.linalg.LinAlgError:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Forgetting factor update
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context

        # Track statistics for adaptive behavior
        self.reward_sums[arm] = self.gamma * self.reward_sums[arm] + reward
        self.counts[arm] += 1

        # Adaptive gamma: if rewards are volatile, increase forgetting
        # (i.e., decrease gamma) to respond faster to changes
        if self.counts[arm] > 10:
            # Estimate recent reward variance
            recent_mean = self.reward_sums[arm] / (1.0 - self.gamma ** self.counts[arm])
            # Simple heuristic: adjust gamma based on reward magnitude
            # Keep gamma in [0.95, 0.999]
            abs_reward = abs(reward)
            if abs_reward > 1.0:
                # High variance signal, forget faster
                target_gamma = 0.98
            else:
                target_gamma = 0.995
            # Smoothly adjust gamma
            self.gamma = 0.99 * self.gamma + 0.01 * target_gamma
            self.gamma = np.clip(self.gamma, 0.95, 0.999)
# EVOLVE-BLOCK-END
