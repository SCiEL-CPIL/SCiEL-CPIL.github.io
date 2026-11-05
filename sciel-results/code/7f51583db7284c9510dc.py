"""Non-stationary linear contextual bandit with Kalman filter and adaptive process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using per-arm Kalman filters:
    - State: theta_a (dim-vector) for each arm
    - Observation model: r = x^T theta_a + noise
    - Adaptive process noise Q driven by innovation (prediction residual)
    - UCB exploration using posterior covariance with count-dampened bonus
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s evaluator limit

        # Kalman filter state per arm
        # theta: state mean (dim,)
        # P: state covariance (dim, dim)
        # Start with high uncertainty (large P) to encourage exploration
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # high initial uncertainty
        self.counts = [0] * n_arms

        # Observation noise variance (fixed, small)
        self.R = 0.1

        # Adaptive process noise parameters
        # Base process noise (small, for slow drift)
        self.Q_base = np.eye(dim) * 0.02
        # Max process noise (for rapid drift detection)
        self.Q_max = np.eye(dim) * 3.0
        # Innovation threshold: when |residual| / sqrt(S) > this, increase Q
        self.innov_threshold = 1.5
        # Adaptation rate for Q
        self.Q_alpha = 0.4

        # UCB exploration coefficient
        self.C = 1.5
        self.C_min = 0.3
        self.C_max = 5.0
        self.count_scale = 0.4  # sqrt(count) dampening of the bonus
        self.warmup_steps = 40

        # Track per-arm innovation statistics for adaptive Q
        self.innov_ema = [0.0] * n_arms  # EMA of squared normalized innovation
        self.innov_ema_alpha = 0.4

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_Q(self, arm, innov_norm_sq):
        """
        Compute adaptive process noise covariance based on innovation.
        innov_norm_sq: (residual^2) / S where S = x^T P x + R
        High innovation -> large Q (forget more)
        Low innovation -> small Q (trust model)
        """
        c = self.counts[arm]
        if c < 3:
            # Early phase: moderate exploration via larger Q
            return self.Q_base * 3.0

        # Use EMA of normalized innovation to smooth Q adaptation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * innov_norm_sq

        # Map innovation EMA to Q scale
        # innov_ema ~ 1.0 (chi-square expectation) -> Q_base
        # innov_ema > 2.0 -> increase Q significantly
        # innov_ema < 0.5 -> decrease Q
        innov_ema = self.innov_ema[arm]

        # Smooth mapping: Q_scale = base * (1 + beta * max(0, innov_ema - threshold))
        # Plus a floor to always have some drift tracking
        excess = max(0.0, innov_ema - self.innov_threshold)
        q_scale = 1.0 + 2.0 * (1.0 - np.exp(-0.8 * excess))
        # Clamp
        q_scale = min(q_scale, 50.0)

        Q = self.Q_base * q_scale
        # Ensure Q doesn't exceed Q_max
        max_eig = np.linalg.norm(Q, 2)
        max_eig_cap = np.linalg.norm(self.Q_max, 2)
        if max_eig > max_eig_cap:
            Q = Q * (max_eig_cap / max_eig)
        return Q

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient based on count and time."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 1.3
        # Gentle decay over time to maintain some exploration under non-stationarity
        time_decay = np.exp(-self.t / 2000.0)
        time_decay = max(0.4, time_decay)
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # UCB bonus: C * sqrt(x^T P x) / (1 + count_scale * sqrt(count))
            c_val = self._adaptive_C(a)
            std = np.sqrt(max(0.0, context @ self.P[a] @ context))
            c = self.counts[a]
            bonus = c_val * std / (1.0 + self.count_scale * np.sqrt(c))
            # Encourage exploring unseen arms early
            if c == 0:
                bonus += 1.0
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1
        self.counts[arm] += 1

        # Kalman filter update for arm
        theta = self.theta[arm]
        P = self.P[arm]

        # Prediction step: add process noise
        # First compute innovation for adaptive Q
        S = context @ P @ context + self.R  # innovation variance
        if S < 1e-12:
            S = 1e-12
        innov = reward - context @ theta  # innovation (residual)
        innov_norm_sq = (innov * innov) / S  # normalized innovation squared

        # Adaptive process noise
        Q = self._adaptive_Q(arm, innov_norm_sq)

        # Predicted covariance (before measurement)
        P_pred = P + Q

        # Kalman gain
        # K = P_pred x^T / S
        P_pred_x = P_pred @ context  # (dim,)
        K = P_pred_x / S  # (dim,)

        # Update state
        self.theta[arm] = theta + K * innov
        # Update covariance: P_new = (I - K x^T) P_pred
        # Use Joseph form for numerical stability:
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        IKxT = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = IKxT @ P_pred @ IKxT.T + (self.R / S) * np.outer(K, K)

        # Periodic numerical stabilization: ensure P remains PSD
        if self.t % 100 == 0:
            for a in range(self.n):
                # Add small ridge to ensure positive definiteness
                self.P[a] += 1e-6 * np.eye(self.d)
                # Symmetrize
                self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
                # Clip diagonal to non-negative
                np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 0.0))
# EVOLVE-BLOCK-END
