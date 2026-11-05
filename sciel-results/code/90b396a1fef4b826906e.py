# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using per-arm Kalman filter
    with adaptive process noise and UCB exploration.
    
    Combines:
    - Joseph-form Kalman updates for numerical stability
    - Innovation-driven adaptive process noise (EMA-based)
    - UCB exploration with adaptive coefficient and count dampening
    - Deadline guard and periodic numerical stabilization
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Kalman filter state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive process noise parameters
        self.q_base = 0.05
        self.q_max = 5.0
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.3
        self.innov_base = 1.0

        # UCB exploration parameters
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov):
        """Compute adaptive process noise scalar based on innovation EMA."""
        c = self.count[arm]
        if c < 3:
            return self.q_base * 2.0

        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        q = min(self.q_max, self.q_base * q_scale)
        return q

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """Kalman update step: rank-1 update with Joseph-form covariance."""
        P = self.P[a]
        theta = self.theta[a]

        pred = float(theta @ ctx)
        innov = reward - pred

        R = 1.0
        var = float(ctx @ P @ ctx) + R + self.jitter

        K = (P @ ctx) / var
        self.theta[a] = theta + (K * innov)

        # Joseph-form covariance update for numerical stability
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T

        # Ensure symmetry and positive semi-definite
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))

        self.count[a] += 1
        return innov

    def _adaptive_u(self):
        """Adapt exploration coefficient over time."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)

        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            var = float(ctx @ self.P[a] @ ctx)
            std = float(np.sqrt(max(var, 0.0)))
            c = self.count[a]
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 2.0
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        pred = float(self.theta[arm] @ ctx)
        innov = reward - pred

        q = self._adaptive_q(arm, innov)
        self._predict(arm, q)
        self._update(arm, ctx, reward)

        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
# EVOLVE-BLOCK-END
