"""Combined approach: Kalman filter with Joseph-form update + time-decaying exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit combining:
    - Kalman filter with Joseph-form covariance update (numerical stability)
    - Tanh-saturated EMA-based adaptive process noise (smooth, bounded Q)
    - Time-decaying exploration (alpha decreases with time, not just count)
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 130.0  # Safety margin before 150s kill

        # Per-arm Kalman state
        # Higher initial P allows faster adaptation to the true theta
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (per-arm) - tanh-saturated EMA
        self.Q_scale = [0.01 for _ in range(n_arms)]
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1  # EMA smoothing factor

        # Exploration: Time-decaying alpha
        # Starts at alpha_base, decays exponentially with time, floors at alpha_min
        self.alpha_base = 1.2
        self.alpha_min = 0.1
        self.decay_rate = 0.01  # exp(-t * decay_rate)

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self):
        """Time-decaying exploration factor."""
        # alpha = alpha_min + (alpha_base - alpha_min) * exp(-t * decay_rate)
        # This ensures we explore heavily at the start and settle into exploitation,
        # but retain a small amount of exploration to catch non-stationary shifts.
        return self.alpha_min + (self.alpha_base - self.alpha_min) * np.exp(-self.t * self.decay_rate)

    def _get_Q(self, arm):
        """Get process noise matrix from scalar scale."""
        return self.Q_scale[arm] * np.eye(self.d)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if time.time() - self._t0 > self._deadline:
            scores = []
            for a in range(self.n):
                mu = self.theta[a] @ context
                scores.append(mu)
            return int(np.argmax(scores))

        # UCB with time-decaying exploration
        alpha_t = self._get_alpha()
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            P_pred = self.P[a] + self._get_Q(a)

            # Compute mean and variance of reward estimate
            mean = self.theta[a] @ context
            var = context @ P_pred @ context
            ucb = mean + alpha_t * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        a = arm
        self.t += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # Predict step: inflate covariance with process noise
        Q_a = self._get_Q(a)
        P_pred = self.P[a] + Q_a

        # Innovation (residual)
        y = reward - self.theta[a] @ context
        S = context @ P_pred @ context + 1.0  # observation noise = 1

        # Kalman gain
        K = P_pred @ context / max(S, 1e-12)

        # Update state
        self.theta[a] = self.theta[a] + K * y

        # Joseph form: P = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        IK = np.eye(self.d) - np.outer(K, context)
        self.P[a] = IK @ P_pred @ IK.T + np.outer(K, K) * 1.0

        # Ensure symmetry and PSD
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        self.P[a] += 1e-8 * np.eye(self.d)

        self.count[a] += 1

        # Adaptive process noise: tanh-saturated EMA of |residual|
        abs_res = abs(y)
        self.residual_ema[a] = (1 - self.ema_alpha) * self.residual_ema[a] + self.ema_alpha * abs_res

        # Map EMA to Q scale using tanh saturation for smooth bounded adaptation
        raw_scale = self.residual_ema[a] * 0.15
        q_scale = 0.001 + 0.499 * np.tanh(raw_scale)
        self.Q_scale[a] = q_scale
# EVOLVE-BLOCK-END
