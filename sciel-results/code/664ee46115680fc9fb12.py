# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Kalman-filter LinUCB for non-stationary linear contextual bandits.
    
    Combines:
    - Kalman filter with adaptive process noise (from current program)
    - Tanh-saturated EMA of squared innovations for smooth Q adaptation (from inspiration 2)
    - Joseph form covariance update for numerical stability (from inspiration 1)
    - Decaying exploration coefficient based on arm count (from inspiration 1)
    - Deadline guard with safety margin
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # seconds, safety margin before 120s kill

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 5e-4
        self._Q_max = 0.05
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # EMA smoothing factor

        # Observation noise
        self._R = 0.1

        # UCB parameter (adaptive)
        self.alpha = 1.0
        self._alpha_min = 0.3
        self._alpha_decay_rate = 300.0

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        # Smooth tanh saturation to avoid extreme Q values
        ema_val = self._innov_ema[arm]
        # Map [0, ~2] to [0, 1] via tanh for smoothness
        scaled = np.tanh(ema_val / 0.5)  # saturates around ema_val ~ 1.5
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def select(self, context):
        """Select arm using UCB with Kalman filter estimates."""
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Fallback: pick arm with highest mean estimate
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted mean
            mu = self.theta[a] @ ctx
            # Predicted variance
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Adaptive exploration: reduce alpha as we learn more about this arm
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            # Also decay alpha globally over time
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
            
            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update Kalman filter for the selected arm with observed reward."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Prediction step (random walk model)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # Update step (measurement: r = theta^T x + noise)
        # Innovation: y = r - theta^T x
        y = r - self.theta[a] @ x

        # Innovation covariance: S = x^T P_pred x + R
        P_pred_x = P_pred @ x
        S = x @ P_pred_x + self._R
        if S < 1e-12:
            S = 1e-12

        # Kalman gain: K = P_pred x / S
        K = P_pred_x / S

        # Update theta
        self.theta[a] = self.theta[a] + K * y

        # Update P (Joseph form for numerical stability)
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)

        # Ensure P is symmetric
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q (EMA of squared innovation)
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        self.count[a] += 1
# EVOLVE-BLOCK-END
