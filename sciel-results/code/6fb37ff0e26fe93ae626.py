"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive Kalman-Filter Linear Bandit with per-arm state tracking."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Per-arm Kalman filter state
        # theta: current weight estimate (d,)
        # P: covariance matrix (d, d)
        # Q: process noise covariance (d, d) - adaptive
        # R: observation noise variance (scalar)

        # Initialize with large uncertainty
        init_P = np.eye(dim) * 10.0
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [init_P.copy() for _ in range(n_arms)]

        # Adaptive process noise
        self.Q_base = np.eye(dim) * 1e-4
        self.R = 1.0  # observation noise variance

        # For adaptive Q: track prediction error
        self.pred_errors = [0.0 for _ in range(n_arms)]
        self.error_history = [np.zeros(20) for _ in range(n_arms)]
        self.error_idx = [0 for _ in range(n_arms)]

        # Exploration parameter
        self.alpha = 0.5

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin

        # Counters
        self.t = 0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64).flatten()

        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Prediction step: account for process noise
            self.P[a] = self.P[a] + self._get_Q(a)

            # Compute score with exploration bonus
            mean = self.theta[a] @ context
            # Uncertainty: sqrt(context^T P context)
            var = context @ self.P[a] @ context
            bonus = self.alpha * np.sqrt(max(var, 0.0))
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        self.t += 1

        # Check deadline
        if time.time() > self.deadline:
            return

        # Track prediction error for adaptive Q
        pred = self.theta[arm] @ context
        error = reward - pred
        self.pred_errors[arm] = error
        self.error_history[arm][self.error_idx[arm]] = error
        self.error_idx[arm] = (self.error_idx[arm] + 1) % len(self.error_history[arm])

        # Adaptive process noise: increase Q when prediction errors are large
        Q = self._get_Q(arm)

        # Kalman update step
        # Observation model: y = x^T theta + noise
        # H = context (1 x d)
        # S = H P H^T + R = context^T P context + R
        S = context @ self.P[arm] @ context + self.R
        # K = P H^T / S = P context / S
        K = self.P[arm] @ context / S
        # Update theta
        self.theta[arm] = self.theta[arm] + K * error
        # Update P: P = (I - K H) P
        I = np.eye(self.d)
        self.P[arm] = (I - np.outer(K, context)) @ self.P[arm]
        # Symmetrize for numerical stability
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

    def _get_Q(self, arm):
        """Adaptive process noise based on recent prediction errors."""
        # Use mean absolute error over recent history
        hist = self.error_history[arm]
        n_used = min(self.t, len(hist))
        if n_used == 0:
            return self.Q_base

        # Scale Q based on recent error magnitude
        recent_errors = hist[:n_used] if self.error_idx[arm] >= n_used else \
                        np.concatenate([hist[self.error_idx[arm]:], hist[:n_used - (len(hist) - self.error_idx[arm])]])
        # Simpler: just use all available entries
        available = hist if self.t >= len(hist) else hist[:self.t]
        mean_abs_err = np.mean(np.abs(available)) if len(available) > 0 else 0.0

        # Adaptive scaling: larger errors -> larger Q
        scale = 1.0 + 5.0 * mean_abs_err
        return self.Q_base * scale
# EVOLVE-BLOCK-END
