# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before the 120 s wall-clock kill


class Policy:
    """Per-arm Kalman filter with adaptive process noise (BOCPD-style reset)."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # state covariance (start large = high uncertainty)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]    # state mean
        self.r2 = 1.0                                           # reward noise variance
        self.base_Q = 1e-4                                      # baseline process noise
        self.max_Q = 5.0                                        # cap on adaptive process noise
        self.alpha = 0.85                                       # EMA smoothing for innovation variance
        self.innov_var = [1.0 for _ in range(n_arms)]           # smoothed innovation variance
        self.age = [0 for _ in range(n_arms)]                   # steps since last reset
        self.reset_age = 120                                    # hard reset after this many steps
        self.deadline = _DEADLINE

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        # Deadline guard: fall back to the best currently-estimated arm.
        if time.time() > self.deadline:
            return int(np.argmax([self.theta[a] @ context for a in range(self.n)]))
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            sigma2 = self.r2 + context @ self.P[a] @ context
            # Adaptive UCB: scale exploration by sqrt of total variance
            scores.append(mu + 1.0 * np.sqrt(sigma2))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        if time.time() > self.deadline:
            return
        a = arm
        x = context
        # Predict: inflate covariance by adaptive process noise
        innov = reward - self.theta[a] @ x
        innov2 = float(innov * innov)
        # Update smoothed innovation variance (detect drift)
        self.innov_var[a] = self.alpha * self.innov_var[a] + (1 - self.alpha) * innov2
        # Adaptive process noise: larger when recent innovations are large (drift)
        Q = self.base_Q + 0.5 * min(self.innov_var[a], self.max_Q)
        self.P[a] = self.P[a] + Q * np.eye(self.d)
        # Measure: Kalman gain
        S = self.r2 + x @ self.P[a] @ x
        K = self.P[a] @ x / S
        # Update mean and covariance
        self.theta[a] = self.theta[a] + K * innov
        I = np.eye(self.d)
        self.P[a] = (I - np.outer(K, x)) @ self.P[a]
        # Symmetrize for numerical stability
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        # Age-based reset for catastrophic drift
        self.age[a] += 1
        if self.age[a] >= self.reset_age:
            self.age[a] = 0
            self.P[a] = np.eye(self.d) * 10.0
            self.innov_var[a] = 1.0
# EVOLVE-BLOCK-END
