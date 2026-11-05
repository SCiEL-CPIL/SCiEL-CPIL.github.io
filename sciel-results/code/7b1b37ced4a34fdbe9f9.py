"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.

    Each arm maintains a state (theta) and covariance (P) updated via a Kalman
    filter with adaptive process noise Q driven by an EMA of squared innovations
    with smooth tanh saturation.  Adaptive UCB alpha decays over time.
    A deadline guard ensures we return within the wall-clock limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # seconds, safety margin before 120 s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 1e-3
        self._Q_max = 0.03
        # EMA of squared innovations per arm (normalized)
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # EMA smoothing factor

        # Observation noise
        self._R = 0.01

        # UCB parameter (adaptive)
        self.alpha = 1.0
        self._alpha_min = 0.3
        self._alpha_decay_rate = 120.0

        # Reward history for safe fallback
        self._reward_ema = [0.0 for _ in range(n_arms)]
        self._reward_alpha = 0.1

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        # Smooth tanh saturation to avoid extreme Q values
        # innov_ema is EMA of (innovation^2 / S) where S is innovation variance
        ema_val = self._innov_ema[arm]
        # Map [0, ~2] to [0, 1] via tanh for smoothness
        scaled = np.tanh(ema_val / 0.5)  # saturates around ema_val ~ 1.5
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest EMA reward
            if any(abs(r) > 1e-10 for r in self._reward_ema):
                return int(np.argmax(self._reward_ema))
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted mean
            mu = self.theta[a] @ ctx
            # Predicted variance
            var = ctx @ self.P[a] @ ctx
            # UCB score
            score = mu + self.alpha * np.sqrt(max(var, 0.0))
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Update reward EMA for fallback
        self._reward_ema[arm] = (1 - self._reward_alpha) * self._reward_ema[arm] + self._reward_alpha * reward

        # Predict step: inflate covariance with adaptive process noise
        q = self._adaptive_Q(arm)
        self.P[arm] = self.P[arm] + q * np.eye(self.d)

        # Update step: Kalman gain
        # Innovation
        pred = self.theta[arm] @ ctx
        innovation = reward - pred
        # Innovation variance
        S = ctx @ self.P[arm] @ ctx + self._R
        # Kalman gain
        K = (self.P[arm] @ ctx) / S
        # State update
        self.theta[arm] = self.theta[arm] + K * innovation
        # Covariance update (Joseph form for numerical stability)
        I_KK = np.eye(self.d) - np.outer(K, ctx)
        self.P[arm] = I_KK @ self.P[arm] @ I_KK.T + q * np.eye(self.d)

        # Symmetrize to prevent drift
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # Update innovation EMA (normalized by S for scale-invariance)
        if self.count[arm] >= 2:
            norm_innov_sq = (innovation * innovation) / max(S, 1e-6)
            self._innov_ema[arm] = (1 - self._innov_alpha) * self._innov_ema[arm] + self._innov_alpha * norm_innov_sq

        self.count[arm] += 1

        # Adaptive exploration: decay alpha over time
        self.alpha = max(self._alpha_min, 1.0 * np.exp(-self.t / self._alpha_decay_rate))
# EVOLVE-BLOCK-END
