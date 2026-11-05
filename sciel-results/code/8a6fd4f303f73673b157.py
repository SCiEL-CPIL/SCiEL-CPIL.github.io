# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit combining:
    - Per-arm Kalman filter with adaptive process noise
    - Prediction-error-driven acceleration of forgetting
    - Huber robustification of the innovation to dampen outlier-driven bias
    - UCB exploration with adaptive coefficient
    - Sliding-window covariance reset for catastrophic shift recovery
    - Deadline guard and periodic numerical stabilization
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # safety margin before 120s kill

        # Kalman filter state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty so unexplored arms are favoured
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive process noise parameters
        self.q_base = 0.05          # base process noise
        self.q_max = 5.0            # cap on adaptive process noise
        self.innov_ema = [0.0] * n_arms  # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.4       # Smoothing factor for EMA
        self.innov_base = 1.0            # Baseline threshold for innovation

        # Context-norm adaptive scaling
        self.ctx_norm_gamma = 0.05
        self.ctx_norm_cap = 3.0

        # Huber robustification
        self.huber_delta = 1.5

        # Sliding-window reset mechanism
        self.reset_threshold_scale = 2.5
        self.reset_cov_scale = 10.0
        self.reset_window = 8
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration multiplier
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.jitter = 1e-6

        # Warmup for exploration boost
        self.step_count = 0
        self.warmup_steps = 20

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov, ctx_norm_sq):
        """
        Compute adaptive process noise scalar based on innovation and context norm.
        Uses EMA of squared innovation to smooth adaptation.
        """
        c = self.count[arm]
        if c < 3:
            # Early phase: moderate exploration via larger Q
            return self.q_base * 2.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        # Map innovation EMA to Q scale
        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 4.0 * (1.0 - np.exp(-1.5 * excess))
        q_scale = min(q_scale, 60.0)

        # Context-norm adaptive scaling
        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        q = min(self.q_max, self.q_base * q_scale * ctx_mult)
        return q

    def _huber_innov(self, innov, std):
        """
        Robustify innovation using Huber loss logic.
        If the innovation is larger than delta * std, clip it to delta * std.
        This prevents single large outliers from dominating the update.
        """
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, q):
        """Kalman predict step: add process noise to covariance."""
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        """
        Kalman update step: rank-1 update with robustified innovation.
        Returns the raw innovation for tracking purposes.
        """
        P = self.P[a]
        theta = self.theta[a]
        pred = float(theta @ ctx)
        innov_raw = reward - pred

        var = float(ctx @ P @ ctx) + 1.0 + self.jitter  # Assuming R=1
        std = float(np.sqrt(var))

        # Robustify innovation
        innov_robust = self._huber_innov(innov_raw, std)

        K = P @ ctx / var
        self.theta[a] = theta + (K * innov_robust)

        # Joseph-form rank-1 update
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T

        # Symmetrise and clip for numerical safety
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        diag = np.diag(self.P[a])
        if np.any(diag < 0) or np.any(diag > 100.0):
            np.fill_diagonal(self.P[a], np.clip(diag, 1e-6, 50.0))
        else:
            np.fill_diagonal(self.P[a], np.maximum(diag, 0.0))

        self.count[a] += 1
        return innov_raw

    def _check_reset(self, a, innov_raw):
        """
        Check if we should reset covariance due to sustained large innovations.
        This handles cases where the filter might be tracking a shifted regime
        poorly and needs a 'hard' restart of uncertainty.
        """
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

        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest current mean estimate (no further exploration).
        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1
        u = self._adaptive_u()

        scores = np.zeros(self.n, dtype=np.float64)
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            std = float(np.sqrt(max(ctx @ self.P[a] @ ctx, 0.0)))
            c = self.count[a]
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 2.0  # encourage exploring unseen arms early
            scores[a] = mean + bonus

        # Soft-max selection: smooth surrogate for argmax that provides
        # gradient-like credit across near-tied arms.
        # Temperature anneals over time to sharpen selection.
        temp = max(0.5, 2.0 * np.exp(-self.step_count / 500.0))
        shifted = scores / temp
        max_s = np.max(shifted)
        exp_s = np.exp(shifted - max_s)
        probs = exp_s / np.sum(exp_s)
        best_arm = int(np.argmax(probs))
        return best_arm

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovation before predict
        pred = float(self.theta[arm] @ ctx)
        innov_raw = reward - pred

        # Context norm squared for adaptive Q
        ctx_norm_sq = float(ctx @ ctx)

        # Adaptive process noise
        q = self._adaptive_q(arm, innov_raw, ctx_norm_sq)

        # Kalman predict step
        self._predict(arm, q)

        # Kalman update step (uses robustified innovation internally)
        self._update(arm, ctx, reward)

        # Check for catastrophic shift and reset if needed
        self._check_reset(arm, innov_raw)

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                P = self.P[a]
                if not np.all(np.isfinite(P)):
                    self.P[a] = np.eye(self.d) * 10.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
                    self.consecutive_large_innov[a] = 0
# EVOLVE-BLOCK-END
