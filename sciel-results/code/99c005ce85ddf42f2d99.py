"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Kalman-filter-based linear bandit with adaptive process noise,
    periodic soft resets, and Thompson Sampling for exploration.
    Designed for non-stationary linear Gaussian contextual bandits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = self._t0 + 115.0

        # Per-arm Kalman filter state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]  # large initial covariance
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_Q = 0.01
        self.Q_min = 0.001
        self.Q_max = 0.1
        self.R = 1.0  # observation noise
        self._innov_ema = [1.0] * n_arms
        self._innov_alpha = 0.1
        self._reset_threshold = 5.0
        self._reset_interval = 50

        # Global time counter
        self.t = 0

        # Deadline guard
        self._start_time = time.time()

        # RNG for Thompson Sampling
        self._rng = np.random.default_rng(seed=42)

    def _check_deadline(self):
        return (time.time() - self._start_time) > self._deadline

    def _adaptive_Q(self, arm):
        """Adapt process noise based on innovation magnitude."""
        ema = self._innov_ema[arm]
        # Higher innovation -> larger process noise (more non-stationarity)
        q = self.base_Q * (1.0 + 0.5 * min(ema, 5.0))
        return max(self.Q_min, min(self.Q_max, q))

    def _maybe_reset(self, arm, innov_sq):
        """Soft reset if innovation is very large or periodically."""
        if innov_sq > self._reset_threshold:
            # Soft reset: inflate covariance
            self.P[arm] = self.P[arm] * 5.0 + np.eye(self.d)
        elif self.t % self._reset_interval == 0 and self.counts[arm] > 0:
            # Periodic gentle inflation to handle slow drift
            self.P[arm] = self.P[arm] * 1.2 + 0.1 * np.eye(self.d)

    def select(self, context):
        if self._check_deadline():
            # Exploit only: pick arm with highest mean estimate
            best_arm = 0
            best_score = -np.inf
            x = np.asarray(context, dtype=np.float64)
            for a in range(self.n):
                score = self.theta[a] @ x
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        x = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Sample from posterior for Thompson Sampling
            theta_sample = self.theta[a] + self._rng.multivariate_normal(
                mean=np.zeros(self.d),
                cov=self.P[a],
                size=1
            )[0]
            score = theta_sample @ x
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = float(reward)

        # Deadline guard
        if self._check_deadline():
            self.counts[a] += 1
            return

        # Prediction step (random walk model)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # Update step (Kalman update)
        y = r - self.theta[a] @ x  # innovation
        P_pred_x = P_pred @ x
        S = x @ P_pred_x + self.R  # innovation variance
        if S < 1e-12:
            S = 1e-12

        K = P_pred_x / S  # Kalman gain

        self.theta[a] = self.theta[a] + K * y
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)
        # Symmetrize for numerical stability
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q
        innov_sq = (y * y) / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.counts[a] += 1
# EVOLVE-BLOCK-END
