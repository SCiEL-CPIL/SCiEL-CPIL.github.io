# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Combines:
    - Joseph-form covariance update for numerical stability
    - Tanh-saturated EMA-based adaptive process noise (smooth, bounded Q)
    - Count-based exploration decay (alpha decreases with arm count)
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._t0 = time.time()
        self._deadline = 130.0  # safety margin before 150s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise (per-arm) - tanh-saturated EMA
        self.Q_scale = [0.01 for _ in range(n_arms)]  # scalar scale for Q
        self.residual_ema = [0.0 for _ in range(n_arms)]  # EMA of |residual|
        self.ema_alpha = 0.1  # EMA smoothing factor

        # Exploration: base alpha with count-based decay
        self.alpha_base = 1.2
        self.alpha_decay = 20.0  # alpha = base / (1 + count/decay)

        # Best arm tracking for deadline fallback
        self.best_arm = 0
        self.best_reward = -np.inf

    def _get_alpha(self, arm):
        """Count-based exploration decay."""
        c = self.count[arm]
        return self.alpha_base / (1.0 + c / self.alpha_decay)

    def _get_Q(self, arm):
        """Get process noise matrix from scalar scale."""
        return self.Q_scale[arm] * np.eye(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
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
        a = arm
        self.t += 1

        # Track best reward for deadline fallback
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_arm = a

        # Predict step
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

        # Ensure symmetry and PSD (lighter than full eigendecomposition)
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        # Add small ridge to ensure PSD without full eigendecomposition
        self.P[a] += 1e-8 * np.eye(self.d)

        self.count[a] += 1

        # Adaptive process noise: tanh-saturated EMA of |residual|
        abs_res = abs(y)
        # Update EMA
        self.residual_ema[a] = (1 - self.ema_alpha) * self.residual_ema[a] + self.ema_alpha * abs_res

        # Map EMA to Q scale using tanh saturation for smooth bounded adaptation
        # Scale: when residual_ema is small -> Q small; when large -> Q saturates
        raw_scale = self.residual_ema[a] * 0.15
        q_scale = 0.001 + 0.499 * np.tanh(raw_scale)  # bounded in [0.001, 0.5]
        self.Q_scale[a] = q_scale
# EVOLVE-BLOCK-END
