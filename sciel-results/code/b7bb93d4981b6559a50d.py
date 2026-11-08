"""Non-stationary linear contextual bandit with adaptive forgetting and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive non-stationary linear bandit with dynamic forgetting factor."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 145.0  # safety margin before 150s kill

        # Per-arm ridge regression with forgetting
        # A[a] = sum of outer(x,x) * decay^(t - t_i)
        # b[a] = sum of r_i * x_i * decay^(t - t_i)
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting: start moderately aggressive, adapt based on performance
        self.forgetting = 0.95
        self.min_forgetting = 0.80
        self.max_forgetting = 0.995

        # Track recent rewards per arm to detect non-stationarity
        self.recent_rewards = {a: [] for a in range(n_arms)}
        self.reward_window = 20

        # UCB parameter
        self.alpha = 1.0

        # Track cumulative reward for adaptation
        self.total_reward = 0.0
        self.recent_total_rewards = []
        self.reward_hist_window = 50

    def _check_deadline(self):
        return (time.time() - self._start_time) > self._deadline

    def _adapt_forgetting(self):
        """Adapt forgetting factor based on recent performance trends."""
        if len(self.recent_total_rewards) < self.reward_hist_window:
            return

        recent = self.recent_total_rewards[-self.reward_hist_window:]
        if len(recent) < 10:
            return

        # Compare recent half vs older half
        half = len(recent) // 2
        recent_mean = np.mean(recent[half:])
        older_mean = np.mean(recent[:half])

        # If performance is dropping, increase exploration (decrease forgetting)
        # If performance is stable/improving, allow more exploitation (increase forgetting)
        if recent_mean < older_mean - 0.01:
            # Performance declining: be more adaptive
            self.forgetting = max(self.min_forgetting, self.forgetting * 0.98)
        elif recent_mean > older_mean + 0.01:
            # Performance improving: be more stable
            self.forgetting = min(self.max_forgetting, self.forgetting * 1.01)

    def select(self, context):
        if self._check_deadline():
            # Fallback: return best arm based on current estimates
            context = np.asarray(context, dtype=np.float64)
            best_arm = 0
            best_score = -1e18
            for a in range(self.n):
                try:
                    inv = np.linalg.inv(self.A[a])
                    theta = inv @ self.b[a]
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return best_arm

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mean = theta @ context
                variance = context @ inv @ context
                ucb = mean + self.alpha * np.sqrt(variance)
                scores.append(ucb)
            except np.linalg.LinAlgError:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Apply forgetting to all arms
        f = self.forgetting
        for a in range(self.n):
            self.A[a] *= f
            self.b[a] *= f

        # Update the chosen arm
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Track rewards for adaptation
        self.total_reward += reward
        self.recent_total_rewards.append(reward)
        if len(self.recent_total_rewards) > self.reward_hist_window * 2:
            self.recent_total_rewards = self.recent_total_rewards[-self.reward_hist_window * 2:]

        # Track per-arm recent rewards
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.reward_window:
            self.recent_rewards[arm] = self.recent_rewards[arm][-self.reward_window:]

        # Adapt forgetting factor periodically
        if self.t % 10 == 0:
            self._adapt_forgetting()
# EVOLVE-BLOCK-END
