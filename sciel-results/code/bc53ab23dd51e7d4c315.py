"""
Combined Policy: Non-stationary Linear Contextual Bandit
Per-arm Kalman filter with adaptive process noise (Q), conservative
adaptive observation noise (R), Huber robustification, and UCB exploration.
The R adaptation is tuned to be a safety valve for regime shifts,
not the primary robustification mechanism (Huber handles that).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a per-arm Kalman filter
    with adaptive process noise, conservative adaptive observation noise,
    Huber robustification, and UCB exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

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

        # Context-norm adaptive scaling
        self.ctx_norm_gamma = 0.05
        self.ctx_norm_cap = 3.0

        # Huber robustification (primary robustification)
        self.huber_delta = 2.0

        # Conservative adaptive observation noise (safety valve only)
        self.r_base = 1.0
        self.r_max = 50.0
        self.r_threshold = 3.0      # Only kick in when |innov|/std > 3 (more conservative)
        self.r_scale = 2.0          # Gentler quadratic scaling

        # Sliding-window reset mechanism
        self.reset_threshold_scale = 2.5
        self.reset_cov_scale = 10.0
        self.reset_window = 8
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration parameters
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov, ctx_norm_sq):
        """Adaptive process noise based on innovation EMA and context norm."""
        c = self.count[arm]
        if c < 3:
            return self.q_base * 2.0

        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        q = min(self.q_max, self.q_base * q_scale * ctx_mult)
        return q

    def _adaptive_r(self, arm, innov, std):
        """
        Conservative adaptive observation noise.
        Only activates for very large innovations (ratio > r_threshold),
        acting as a safety valve against catastrophic regime shifts.
        """
        if std <= 0:
            return self.r_base

        ratio = abs(innov) / std
        if ratio <= self.r_threshold:
            return self.r_base

        # Gentle quadratic scaling beyond threshold
        excess = ratio - self.r_threshold
        r = self.r_base * (1.0 + self.r_scale * excess * excess)
        return min(r, self.r_max)

    def _huber_innov(self, innov, std):
        """Huber-loss based innovation clipping (primary robustification)."""
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, q):
        """Kalman predict step."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """Kalman update with conservative adaptive R and Huber clipping."""
        P = self.P[a]
        theta = self.theta[a]

        pred = float(theta @ ctx)
        innov_raw = reward - pred

        # Base innovation variance
        var_base = float(ctx @ P @ ctx) + self.r_base + self.jitter
        std_base = float(np.sqrt(var_base))

        # Conservative adaptive R (only for very large innovations)
        R = self._adaptive_r(a, innov_raw, std_base)

        # Final innovation variance
        var = float(ctx @ P @ ctx) + R + self.jitter
        std_final = float(np.sqrt(var))

        # Huber clipping based on base std (conservative, before R inflation)
        innov_robust = self._huber_innov(innov_raw, std_base)

        # Kalman gain
        K = (P @ ctx) / var

        # Update mean
        self.theta[a] = theta + (K * innov_robust)

        # Joseph-form covariance update
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T

        # Ensure symmetry
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)

        # Stability check
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))

        self.count[a] += 1
        return innov_raw

    def _check_reset(self, a, innov_raw):
        """Covariance reset for sustained large innovations."""
        P = self.P[a]
        var = float(np.trace(P)) / self.d
        std = float(np.sqrt(max(var, 0.0)))
        threshold = self.reset_threshold_scale * std + 0.5

        if abs(innov_raw) > threshold:
            self.consecutive_large_innov[a] += 1
        else:
            self.consecutive_large_innov[a] = 0

        if self.consecutive_large_innov[a] >= self.reset_window:
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
            return True
        return False

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
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

        ctx_norm_sq = float(ctx @ ctx)
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred

        q = self._adaptive_q(arm, innov_raw, ctx_norm_sq)
        self._predict(arm, q)
        self._update(arm, ctx, reward)
        self._check_reset(arm, innov_raw)

        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                P = 0.5 * (P + P.T)
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
                    self.consecutive_large_innov[a] = 0
# EVOLVE-BLOCK-END
