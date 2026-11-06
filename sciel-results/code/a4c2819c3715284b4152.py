"""
Non-stationary linear contextual bandit policy using a Kalman filter with
adaptive process noise. This is a structural change from the per-arm RLS
approach: instead of maintaining separate RLS estimators per arm, we use a
single Kalman filter per arm with an adaptive process noise covariance that
increases when prediction errors are large (indicating regime change) and
decreases when errors are small (indicating stable regime).

Key structural differences from RLS:
1. Kalman filter formulation with explicit state transition (identity) and
   adaptive process noise Q_t.
2. The process noise is driven by a robust estimate of recent squared errors,
   allowing the filter to "reset" its belief when the environment changes.
3. Exploration uses a UCB based on the posterior covariance of the Kalman
   filter, which naturally shrinks as more data is observed.
4. No forgetting factor lambda; instead, the process noise directly controls
   how much past information is retained.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Kalman filter state per arm:
        # theta: weight vector
        # P: covariance matrix of theta
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]  # High initial uncertainty

        # Adaptive process noise parameters
        self.Q_base = 1e-4
        self.Q_max = 1.0
        self.error_ema = [0.0] * n_arms
        self.error_ema_alpha = 0.2
        self.error_scale = [1.0] * n_arms
        self.recent_errors = [[] for _ in range(n_arms)]
        self.error_window = 50

        # Exploration
        self.alpha = 1.0
        self.total_rounds = 0
        self.arm_counts = [0] * n_arms

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 145.0

    def _compute_process_noise(self, arm):
        """
        Compute adaptive process noise based on recent prediction errors.
        Large errors -> large Q (fast adaptation)
        Small errors -> small Q (stable tracking)
        """
        # Use robust scale: median absolute deviation of recent errors
        errs = self.recent_errors[arm]
        if len(errs) >= 10:
            arr = np.array(errs)
            mad = np.median(np.abs(arr - np.median(arr)))
            scale = max(mad, 1e-4)
        else:
            scale = self.error_scale[arm]

        # Normalize error by scale
        ema_err = self.error_ema[arm]
        normalized = ema_err / (scale + 1e-8)

        # Map normalized error to Q in [Q_base, Q_max] using sigmoid
        k = 5.0
        x0 = 0.5
        t = 1.0 / (1.0 + np.exp(-k * (normalized - x0)))
        Q = self.Q_base + t * (self.Q_max - self.Q_base)
        return Q

    def _update_error_tracking(self, arm, error):
        """Update error tracking for adaptive process noise."""
        sq_err = error * error
        self.error_ema[arm] = (1.0 - self.error_ema_alpha) * self.error_ema[arm] + self.error_ema_alpha * sq_err
        self.recent_errors[arm].append(error)
        if len(self.recent_errors[arm]) > self.error_window:
            self.recent_errors[arm].pop(0)
        # Update robust scale
        if len(self.recent_errors[arm]) >= 10:
            arr = np.array(self.recent_errors[arm])
            mad = np.median(np.abs(arr - np.median(arr)))
            self.error_scale[arm] = 0.9 * self.error_scale[arm] + 0.1 * max(mad, 1e-4)

    def _get_alpha(self):
        """Adaptive exploration parameter with decay."""
        if self.total_rounds < 20:
            return self.alpha + 1.5
        decay = 1.0 / (1.0 + self.total_rounds / 150.0)
        return self.alpha * decay + 0.3

    def select(self, context):
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        alpha = self._get_alpha()
        scores = []

        for a in range(self.n):
            mu = self.theta[a] @ context
            var = context @ (self.P[a] @ context)
            std = np.sqrt(max(0.0, var))
            score = mu + alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self._deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        self.total_rounds += 1
        self.arm_counts[arm] += 1

        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred

        # Update error tracking
        self._update_error_tracking(arm, error)

        # Compute adaptive process noise
        Q = self._compute_process_noise(arm)
        Q_mat = Q * np.eye(self.d)

        # Kalman filter update:
        # Prediction step (identity transition):
        # theta_pred = theta
        # P_pred = P + Q

        P_pred = self.P[arm] + Q_mat
        P_pred = (P_pred + P_pred.T) * 0.5

        # Update step:
        # Innovation: v = reward - H @ theta_pred
        # Innovation covariance: S = H @ P_pred @ H^T + R
        # Kalman gain: K = P_pred @ H^T @ S^{-1}
        # theta = theta_pred + K @ v
        # P = (I - K @ H) @ P_pred

        H = context  # 1 x d
        H_row = context.reshape(1, -1)  # 1 x d

        # S = H @ P_pred @ H^T + R, where R is observation noise
        R = 1.0
        S = float(H_row @ P_pred @ H_row.T) + R
        S = max(S, 1e-12)

        # Kalman gain: K = P_pred @ H^T / S
        K = (P_pred @ H_row.T) / S  # d x 1

        # Update theta
        self.theta[arm] = self.theta[arm] + K * error

        # Update P: P = (I - K @ H) @ P_pred
        I = np.eye(self.d)
        KH = np.outer(K, H_row)  # d x 1 @ 1 x d = d x d
        self.P[arm] = (I - KH) @ P_pred
        self.P[arm] = (self.P[arm] + self.P[arm].T) * 0.5

        # Ensure positive semi-definite
        self.P[arm] += 1e-8 * np.eye(self.d)
