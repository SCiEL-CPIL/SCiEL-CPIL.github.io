"""Non-stationary linear contextual bandit policy with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary LinUCB with:
    - Per-arm adaptive forgetting factor based on reward deviation from EMA
    - Per-arm exponential moving average for baseline detection
    - Adaptive exploration parameter
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Core matrices for LinUCB
        # Use moderate initial regularization for stability
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor: per-arm adaptive
        self.base_gamma = 0.99
        self.gamma = [0.99 for _ in range(n_arms)]

        # Exploration parameter: adaptive
        self.alpha = 1.8
        self.alpha_min = 0.8
        self.alpha_max = 3.0

        # Tracking for non-stationarity detection
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history_len = 30

        # Per-arm EMA of rewards to detect shifts - faster adaptation
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_ema_alpha = 0.5

        # Per-arm deviation tracking
        self.reward_deviation = [0.0 for _ in range(n_arms)]

        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit

        # Best known mean reward for early termination/safety
        self.best_reward_seen = 0.0

        # Track per-arm visit counts for adaptive alpha
        self.arm_visits = [0] * n_arms

        # Track recent reward variance per arm for adaptive alpha
        self.reward_variance = [0.0] * n_arms

    def select(self, context):
        self.t += 1

        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return best-known arm based on highest EMA
            best_arm = int(np.argmax(self.reward_ema))
            return best_arm

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        eye_d = np.eye(self.d)

        # Adaptive alpha: reduce exploration as we gain confidence
        # Scale alpha inversely with sqrt of total observations
        total_obs = sum(self.arm_visits)
        if total_obs > 0:
            self.alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * np.log1p(total_obs) / np.log1p(1000)
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

        for a in range(self.n):
            # Regularize A to ensure invertibility
            A = self.A[a] + eye_d * 1e-6
            try:
                # Use solve instead of explicit inverse for efficiency
                theta = np.linalg.solve(A, self.b[a])
                # For uncertainty, we need A^{-1} @ context
                Ax = np.linalg.solve(A, context)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                Ax = context

            mean = theta @ context
            # Uncertainty: sqrt(x^T A^{-1} x) = sqrt(x @ Ax)
            unc = np.sqrt(max(0.0, context @ Ax))
            scores[a] = mean + self.alpha * unc

        # Clip to reasonable range to avoid extreme values
        scores = np.clip(scores, -10.0, 10.0)

        # Small random tie-breaking for stability
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update visit count
        self.arm_visits[arm] += 1

        # Update EMA with faster adaptation
        if len(self.reward_history[arm]) > 0:
            self.reward_ema[arm] = self.reward_ema_alpha * reward + (1 - self.reward_ema_alpha) * self.reward_ema[arm]
        else:
            self.reward_ema[arm] = reward

        # Update reward variance estimate
        if len(self.reward_history[arm]) > 1:
            recent_rewards = self.reward_history[arm][-10:]
            self.reward_variance[arm] = np.var(recent_rewards)
        else:
            self.reward_variance[arm] = 0.0

        # Per-arm adaptive forgetting based on deviation from EMA
        deviation = abs(reward - self.reward_ema[arm])
        # Smoothly track deviation
        self.reward_deviation[arm] = 0.7 * self.reward_deviation[arm] + 0.3 * deviation

        # Map deviation to gamma: higher deviation -> lower gamma (faster forgetting)
        # deviation ~0 -> gamma ~0.999, deviation > 1 -> gamma ~0.90
        dev_scaled = min(self.reward_deviation[arm] / 1.0, 1.0)
        gamma_target = 0.999 - 0.099 * dev_scaled
        gamma_target = max(0.85, min(0.999, gamma_target))
        # Smoothly adapt per-arm gamma
        self.gamma[arm] = 0.8 * self.gamma[arm] + 0.2 * gamma_target

        # Apply per-arm forgetting and update
        gamma = self.gamma[arm]
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Maintain reward history
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history_len:
            self.reward_history[arm].pop(0)

        # Track best reward
        if reward > self.best_reward_seen:
            self.best_reward_seen = reward

        # Periodic renormalization to prevent numerical issues
        if self.t % 50 == 0:
            for a in range(self.n):
                # Ensure A stays well-conditioned
                try:
                    eigvals = np.linalg.eigvalsh(self.A[a])
                    if np.min(eigvals) < 1e-4:
                        self.A[a] += np.eye(self.d) * 1e-4
                except np.linalg.LinAlgError:
                    self.A[a] = np.eye(self.d) * 0.1
# EVOLVE-BLOCK-END
