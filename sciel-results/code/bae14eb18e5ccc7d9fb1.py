"""Combined approach: Kalman filter with Joseph-form update + adaptive forgetting from ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit combining:
    - Kalman filter with Joseph-form covariance update (from inspirations)
    - Adaptive forgetting via exponential decay (from current program)
    - Tanh-saturated EMA-based adaptive process noise (from inspirations)
    - Count-based exploration decay (from inspirations)
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 135.0  # Safety margin before 150s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive process noise (per-arm) - tanh-saturated EMA
        self.Q_scale = [0.01 for _ in range(n_arms)]
        self.residual_ema = [0.0 for _ in range(n_arms)]
        self.ema_alpha = 0.1

        # Adaptive forgetting (from current program's approach)
        self.prediction_errors = [[] for _ in range(n_arms)]
        self.max_err_window = 50
        self.base_gamma = 0.93
        self.gamma_min = 0.84
        self.gamma_max = 0.99

        # Exploration: count-based decay
        self.alpha_base = 1.5
        self.alpha_decay = 25.0

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor using smooth log-ratio of recent vs older errors."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        recent_errs = self.prediction_errors[arm]
        if len(recent_errs) < 10:
            return self.base_gamma

        n = len(recent_errs)
        recent_mean = np.mean(recent_errs[-10:])
        older_mean = np.mean(recent_errs[:max(1, n - 10)])

        if older_mean < 1e-12:
            log_ratio = 0.0
        else:
            ratio = recent_mean / older_mean
            log_ratio = np.log(max(ratio, 1e-6))

        adjustment = -1.8 * log_ratio
        gamma = self.base_gamma + adjustment

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _get_alpha(self, arm):
        """Count-based exploration decay."""
        c = self.counts[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def _get_Q(self, arm):
        """Get process noise matrix from scalar scale."""
        return self.Q_scale[arm] * np.eye(self.d)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                mu = self.theta[a] @ context
                scores.append(mu)
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            P_pred = self.P[a] + self._get_Q(a)

            # Compute mean and variance of reward estimate
            mean = self.theta[a] @ context
            var = context @ P_pred @ context
            alpha_a = self._get_alpha(a)
            ucb = mean + alpha_a * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        arm = int(np.argmax(scores))
        self.best_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        a = arm

        # Track statistics
        self.counts[a] += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # Compute prediction error before updating
        pred = self.theta[a] @ context
        err = abs(reward - pred)
        self.prediction_errors[a].append(err)
        if len(self.prediction_errors[a]) > self.max_err_window:
            self.prediction_errors[a].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(a)

        # Apply forgetting to covariance
        self.P[a] = gamma * self.P[a]

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

        # Adaptive process noise: tanh-saturated EMA of |residual|
        abs_res = abs(y)
        self.residual_ema[a] = (1 - self.ema_alpha) * self.residual_ema[a] + self.ema_alpha * abs_res

        # Map EMA to Q scale using tanh saturation for smooth bounded adaptation
        raw_scale = self.residual_ema[a] * 0.15
        q_scale = 0.001 + 0.499 * np.tanh(raw_scale)
        self.Q_scale[a] = q_scale
# EVOLVE-BLOCK-END
