"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting factor and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill
        self.best_arm = 0
        self.arm_rewards = [0.0] * n_arms
        self.arm_counts = [0] * n_arms
        self.forgetting = 0.99
        self.alpha = 0.5
        self._last_update_time = time.time()

    def select(self, context):
        # Deadline guard: if we're close to the limit, return the best known arm
        elapsed = time.time() - self.start_time
        if elapsed > self.deadline:
            # Return the arm with highest average reward so far
            if self.arm_counts[0] > 0:
                avgs = [self.arm_rewards[a] / max(self.arm_counts[a], 1) for a in range(self.n)]
                return int(np.argmax(avgs))
            return self.best_arm

        self.t += 1

        # Adaptive exploration: start with more exploration, decrease over time
        # Use a decay schedule for alpha
        decay = max(0.1, 1.0 / (1.0 + self.t / 100.0))
        current_alpha = self.alpha * decay

        scores = []
        for a in range(self.n):
            # Compute theta and uncertainty
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mu = theta @ context
                # Uncertainty term
                v = context @ inv @ context
                ucb = mu + current_alpha * np.sqrt(max(v, 0))
            except np.linalg.LinAlgError:
                mu = 0.0
                ucb = 1.0  # High uncertainty if matrix is singular

            scores.append(ucb)

        # Add small random noise for tie-breaking and exploration
        noise = np.random.randn(self.n) * 0.01
        scores = np.array(scores) + noise

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1

        # Update statistics
        self.arm_rewards[arm] += reward
        self.arm_counts[arm] += 1
        self.best_arm = arm

        # Adaptive forgetting factor: increase forgetting rate if rewards are changing
        # Simple heuristic: if recent reward differs significantly from average, increase forgetting
        if self.arm_counts[arm] > 10:
            avg_reward = self.arm_rewards[arm] / self.arm_counts[arm]
            diff = abs(reward - avg_reward)
            # Increase forgetting if variance is high
            if diff > 0.5:
                self.forgetting = min(0.999, self.forgetting + 0.001)
            else:
                self.forgetting = max(0.98, self.forgetting - 0.0005)

        # Update with forgetting factor
        self.A[arm] = self.forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forgetting * self.b[arm] + reward * context

        # Periodically re-normalize to prevent numerical issues
        if self.t % 100 == 0:
            for a in range(self.n):
                self.A[a] = self.A[a] / (1.0 + 1e-6)
# EVOLVE-BLOCK-END
