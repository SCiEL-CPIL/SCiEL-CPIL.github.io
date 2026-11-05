# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Dual-timescale Kalman filter for non-stationary linear contextual bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Kalman filter parameters
        self.R = 1.0  # observation noise
        self.Q_base = 1e-4  # base process noise for random walk
        self.Q_adapt = 0.01  # adaptive process noise scaling
        self.P_init = np.eye(dim) * 10.0  # initial covariance
        self.alpha = 0.95  # smoothing for error tracking
        self.gamma = 0.5  # exploration bonus coefficient

        # Per-arm Kalman state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]  # weight estimates
        self.P = [self.P_init.copy() for _ in range(n_arms)]  # covariance matrices
        self.P_inv = [np.linalg.inv(self.P_init) for _ in range(n_arms)]  # inverse covariance

        # Adaptive process noise tracking
        self.pred_err = [0.0 for _ in range(n_arms)]  # smoothed prediction error
        self.last_update = [0 for _ in range(n_arms)]  # last update time
        self.update_count = [0 for _ in range(n_arms)]  # number of updates

        # Time tracking for deadline
        self.t0 = time.time()
        self.deadline = 145.0  # safety margin before 150s

    def select(self, context):
        # Deadline guard
        if time.time() - self.t0 > self.deadline:
            return int(np.argmax([t @ context for t in self.theta]))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            P = self.P[a]
            # Prediction: mean and variance
            mean = theta @ context
            var = context @ P @ context
            # UCB score with adaptive exploration
            bonus = self.gamma * np.sqrt(var)
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.t0 > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        a = arm
        self.update_count[a] += 1

        # Predict step: inflate covariance with adaptive process noise
        # Adaptive Q based on recent prediction error
        pred_err_smoothed = self.pred_err[a]
        # Scale process noise by ratio of prediction error to observation noise
        Q_scale = 1.0 + self.Q_adapt * np.abs(pred_err_smoothed) / self.R
        Q = self.Q_base * Q_scale * np.eye(self.dim)

        # Update covariance: P = P + Q (random walk model)
        self.P[a] = self.P[a] + Q
        self.P_inv[a] = np.linalg.inv(self.P[a])

        # Update step: Kalman gain and state update
        # K = P * x^T * (x^T * P * x + R)^{-1}
        Px = self.P[a] @ context
        denom = context @ Px + self.R
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom  # gain vector

        # Prediction error
        pred = self.theta[a] @ context
        err = reward - pred
        self.pred_err[a] = self.alpha * self.pred_err[a] + (1 - self.alpha) * err

        # Update state: theta = theta + K * err
        self.theta[a] = self.theta[a] + K * err

        # Update covariance using Joseph form for numerical stability
        # P = (I - K * x^T) * P * (I - K * x^T)^T + K * R * K^T
        I_KxT = np.eye(self.dim) - np.outer(K, context)
        self.P[a] = I_KxT @ self.P[a] @ I_KxT.T + K * self.R * K.T
        # Symmetrize
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0
        self.P_inv[a] = np.linalg.inv(self.P[a])
# EVOLVE-BLOCK-END
