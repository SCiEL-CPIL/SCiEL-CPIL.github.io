"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with per-arm forgetting, solve-based computation, and drift detection."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        # Larger initial regularization to reduce excessive early exploration
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.count = [0] * n_arms
        self.total_reward = [0.0] * n_arms
        self.last_reward = [0.0] * n_arms
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.forgetting = 0.98
        self.base_alpha = 1.0
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin before 150s kill

    def _solve(self, A, b):
        """Solve A x = b efficiently with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _select_arm(self, context):
        """Select arm using LinUCB with solve-based computation."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adaptive alpha: start with higher exploration, decay over time
        alpha = self.base_alpha * max(0.5, 1.0 - 0.3 * min(1.0, self.t / 200.0))

        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            theta = self._solve(A, b)
            mu = theta @ context

            # Compute variance: context^T A^{-1} context = context^T (A^{-1} context)
            # Use solve to avoid explicit inverse
            try:
                A_inv_context = np.linalg.solve(A, context)
                v = context @ A_inv_context
            except np.linalg.LinAlgError:
                v = 1.0

            # Per-arm adaptive alpha: arms with few pulls get more exploration
            alpha_a = alpha * (1.0 + 0.5 / (1.0 + self.count[a]))
            scores.append(mu + alpha_a * np.sqrt(max(v, 0.0)))

        return int(np.argmax(scores))

    def _detect_drift(self, arm, reward):
        """Detect if rewards for an arm are changing rapidly."""
        recent = self.recent_rewards[arm]
        if len(recent) < 3:
            return False
        # Compare most recent reward to average of previous few
        prev_avg = np.mean(recent[-5:-1]) if len(recent) >= 5 else np.mean(recent[:-1])
        # If recent reward deviates significantly from recent average, flag drift
        return abs(reward - prev_avg) > 0.5 * (1.0 + abs(prev_avg))

    def select(self, context):
        # Deadline guard: if we're close to the limit, return a reasonable default
        if time.time() > self.deadline:
            # Return the arm with highest historical average reward
            if any(self.count[a] > 0 for a in range(self.n)):
                avgs = [self.total_reward[a] / max(self.count[a], 1) for a in range(self.n)]
                return int(np.argmax(avgs))
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Detect drift before updating
        drift = self._detect_drift(arm, reward)

        # Adaptive forgetting: increase forgetting rate if drift is detected
        if drift:
            self.forgetting = min(0.95, self.forgetting * 0.95)
        else:
            # Slowly recover forgetting factor toward baseline
            self.forgetting = 0.98 + 0.02 * (self.forgetting - 0.98)
            self.forgetting = max(0.95, min(0.99, self.forgetting))

        # Update with forgetting factor
        self.A[arm] = self.forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forgetting * self.b[arm] + reward * context

        self.count[arm] += 1
        self.total_reward[arm] = self.forgetting * self.total_reward[arm] + reward
        self.last_reward[arm] = reward

        # Track recent rewards for drift detection
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > 10:
            self.recent_rewards[arm].pop(0)


# EVOLVE-BLOCK-END
