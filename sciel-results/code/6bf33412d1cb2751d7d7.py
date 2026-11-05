"""Bootstrap spawn: direct approach for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # For adaptive forgetting
        self.prediction_errors = [[] for _ in range(n_arms)]
        self.max_err_window = 50

        # Exploration
        self.C = 2.0
        self.base_gamma = 0.93
        self.gamma_min = 0.82
        self.gamma_max = 0.985

        # Track last few contexts for possible restarts
        self.recent_contexts = []
        self.max_recent = 50

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction errors."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # Use recent prediction errors to detect non-stationarity
        recent_errs = self.prediction_errors[arm]
        if len(recent_errs) < 5:
            return self.base_gamma

        # Compare recent errors to earlier ones
        n = len(recent_errs)
        recent_mean = np.mean(recent_errs[-10:])
        older_mean = np.mean(recent_errs[:max(1, n - 10)])

        if older_mean < 1e-12:
            ratio = 1.0
        else:
            ratio = recent_mean / older_mean

        # Smooth nonlinear mapping: use log-ratio for smoother response
        log_ratio = np.log(max(ratio, 1e-6))
        # Map log_ratio to gamma adjustment
        # log_ratio > 0 means errors increasing -> need faster forgetting (lower gamma)
        # log_ratio < 0 means errors decreasing -> can use slower forgetting (higher gamma)
        # Use a steeper response to react more aggressively to non-stationarity
        adjustment = -2.0 * log_ratio  # more aggressive adaptation
        gamma = self.base_gamma + adjustment

        return max(self.gamma_min, min(self.gamma_max, gamma))

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient based on uncertainty and count."""
        c = self.counts[arm]
        if c < 3:
            return self.C * 2.5
        if c < 8:
            return self.C * 1.8
        if c < 15:
            return self.C * 1.4
        if c < 30:
            return self.C * 1.15
        return self.C

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error before updating
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        self.prediction_errors[arm].append(err)
        if len(self.prediction_errors[arm]) > self.max_err_window:
            self.prediction_errors[arm].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
