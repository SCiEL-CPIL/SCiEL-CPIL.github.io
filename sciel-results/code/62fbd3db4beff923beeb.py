"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s evaluator limit

        # Inverse covariance matrices V = A^{-1} for each arm
        ridge = 0.15
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters — tuned forgetting for drift tracking
        self.base_gamma = 0.94
        self.gamma_min = 0.75
        self.gamma_max = 0.99
        self.C = 1.2
        self.C_min = 0.2
        self.C_max = 4.0

        # For tracking prediction errors to adapt forgetting
        self.pred_err_ema = [0.0] * n_arms
        self.err_ema_alpha = 0.35

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on recent prediction error (EMA)."""
        c = self.counts[arm]
        if c < 3:
            return self.base_gamma

        err_ema = self.pred_err_ema[arm]
        err_scale = min(err_ema, 4.0)
        # Smooth, monotonic mapping with stronger response
        gamma = self.base_gamma - 0.12 * (1.0 - np.exp(-0.7 * err_scale))
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and time with smooth decay."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 8:
            return self.C * 1.5
        # Smooth exponential decay of exploration over time
        time_decay = np.exp(-self.t / 1200.0)
        time_decay = max(0.30, time_decay)
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(max(0.0, context @ self.V[a] @ context))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        # Track statistics
        self.counts[arm] += 1

        # Compute prediction error for adaptive gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context
        err = reward - pred
        # Update EMA of squared error
        self.pred_err_ema[arm] = (1.0 - self.err_ema_alpha) * self.pred_err_ema[arm] + self.err_ema_alpha * (err * err)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d) * (1.0 + 0.1)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Periodic numerical stabilization: every 120 steps, add small ridge
        if self.t % 120 == 0:
            for a in range(self.n):
                self.V[a] += 0.002 * np.eye(self.d)
# EVOLVE-BLOCK-END
