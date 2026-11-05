"""Non-stationary linear contextual bandit with Kalman Filter and adaptive process noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using Kalman Filter per arm.
    Models theta as a random walk: theta_t = theta_{t-1} + w_t, w_t ~ N(0, Q)
    Observation: r_t = theta_t^T x_t + v_t, v_t ~ N(0, R)
    
    Adaptive Q: scale process noise by innovation magnitude to track drift.
    Selection: UCB based on posterior mean + alpha * posterior std.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm state: theta (mean) and P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P to large value (high initial uncertainty)
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # Measurement noise
        self.R = 1.0

        # Base process noise (random walk variance per dimension)
        self.Q_base = 0.05
        self.Q_min = 0.01
        self.Q_max = 1.0

        # UCB exploration coefficient
        self.alpha = 1.0

        # Track innovations for adaptive Q
        self.innovation_history = []
        self.inn_window = 30
        self.base_inn = None

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _check_deadline(self):
        if time.time() - self.start_time > self.deadline:
            return True
        return False

    def _adaptive_Q(self, innovation):
        """Adapt process noise based on innovation magnitude."""
        self.innovation_history.append(abs(innovation))
        if len(self.innovation_history) > self.inn_window:
            self.innovation_history.pop(0)

        if self.base_inn is None:
            if len(self.innovation_history) >= 10:
                self.base_inn = np.mean(self.innovation_history) + 1e-6

        if self.base_inn is not None:
            ratio = abs(innovation) / self.base_inn
            # If innovation is large relative to baseline, increase Q (more drift expected)
            # If innovation is small, decrease Q (stable period)
            if ratio > 2.0:
                self.Q_base = min(self.Q_max, self.Q_base * 1.5)
            elif ratio < 0.5:
                self.Q_base = max(self.Q_min, self.Q_base * 0.85)
            else:
                # Gentle recovery toward base
                self.Q_base = self.Q_base * 0.95 + 0.05 * 0.05
        else:
            # Gradually increase Q as we learn
            self.Q_base = min(self.Q_max, self.Q_base + 0.002)

    def select(self, context):
        """Select arm using UCB with Kalman posterior."""
        if self._check_deadline():
            # Fallback: greedy on mean
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)

        ucb_scores = []
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Posterior std: sqrt(x^T P x)
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 0.0))
            ucb = mean + self.alpha * std
            ucb_scores.append(ucb)

        return int(np.argmax(ucb_scores))

    def update(self, context, arm, reward):
        """Kalman Filter update for the selected arm."""
        context = np.asarray(context, dtype=np.float64)
        x = context
        r = float(reward)

        # --- PREDICTION STEP (random walk) ---
        # theta_pred = theta (no change in mean for random walk)
        # P_pred = P + Q
        Q = self.Q_base * np.eye(self.d)
        P_pred = self.P[arm] + Q

        # --- MEASUREMENT UPDATE ---
        # Innovation: y = r - x^T theta
        pred = self.theta[arm] @ x
        innovation = r - pred

        # Adapt Q based on this innovation (for future updates)
        self._adaptive_Q(innovation)

        # Innovation covariance: S = x^T P_pred x + R
        S = x @ P_pred @ x + self.R

        # Kalman gain: K = P_pred x / S
        K = (P_pred @ x) / S  # shape: (d,)

        # State update: theta = theta + K * innovation
        self.theta[arm] = self.theta[arm] + K * innovation

        # Covariance update: P = (I - K x^T) P_pred
        # P = P_pred - K (x^T P_pred)
        # K (x^T P_pred) = (P_pred x / S) (x^T P_pred) = outer(P_pred @ x, P_pred @ x) / S
        self.P[arm] = P_pred - np.outer(P_pred @ x, P_pred @ x) / S

        # Ensure P remains PSD (numerical stability)
        # Symmetrize
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
# EVOLVE-BLOCK-END
