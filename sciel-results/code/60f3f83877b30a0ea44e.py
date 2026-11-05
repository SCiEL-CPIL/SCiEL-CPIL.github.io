"""Non-stationary linear contextual bandit with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with time-decaying coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Per-arm state
        self.V = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # inverse covariance
        self.b = [np.zeros(dim) for _ in range(n_arms)]       # weighted sum of rewards*x
        self.counts = [0] * n_arms
        self.theta = [np.zeros(dim) for _ in range(n_arms)]   # cached theta = V @ b

        # Global adaptive parameters
        self.gamma = 0.97          # base forgetting factor
        self.gamma_min = 0.85
        self.gamma_max = 0.999
        self.lambda_ridge = 0.1    # ridge regularization

        # Exploration: starts aggressive, decays
        self.C_base = 2.0
        self.C_min = 0.5
        self.C_max = 4.0
        self.round_num = 0

        # Prediction error tracking for adaptive gamma
        self.pred_error_window = []
        self.pred_error_window_max = 30
        self.base_pred_error = None

        # Per-arm recent reward stats for variance-based gamma
        self.arm_reward_sums = [0.0] * n_arms
        self.arm_reward_sq_sums = [0.0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_C(self, arm):
        """Time-decaying exploration coefficient with per-arm boost for low counts."""
        c = self.counts[arm]
        # Global decay: start high, settle to C_min
        t = self.round_num
        decay = 1.0 / (1.0 + t / 50.0)
        C_global = self.C_min + (self.C_max - self.C_min) * decay

        # Per-arm boost for under-explored arms
        if c < 3:
            C_arm = self.C_max
        elif c < 10:
            C_arm = C_global * 1.5
        else:
            C_arm = C_global

        return min(C_arm, self.C_max)

    def _adaptive_gamma(self, arm):
        """
        Adapt forgetting factor based on:
        1. Global prediction error trend (spike -> faster forgetting)
        2. Per-arm reward variance (high variance -> faster forgetting)
        """
        c = self.counts[arm]

        # Start with base gamma
        gamma = self.gamma

        # Global prediction error signal
        if len(self.pred_error_window) >= 5:
            recent_errors = np.array(self.pred_error_window[-10:])
            mean_err = np.mean(recent_errors)
            if self.base_pred_error is not None and self.base_pred_error > 1e-8:
                ratio = mean_err / self.base_pred_error
                # If errors are growing, accelerate forgetting
                if ratio > 1.5:
                    gamma -= 0.05 * min(ratio - 1.0, 2.0)
                elif ratio < 0.7:
                    # Errors shrinking, slow down forgetting
                    gamma += 0.02 * (1.0 - ratio)

        # Per-arm variance signal
        if c >= 5:
            mean_r = self.arm_reward_sums[arm] / c
            var_r = max(0.0, self.arm_reward_sq_sums[arm] / c - mean_r ** 2)
            var_scale = min(var_r, 5.0)
            gamma -= 0.03 * var_scale

        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _update_theta(self, arm):
        """Update cached theta for arm."""
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            c_val = self._get_C(a)
            # UCB bonus: c * sqrt(x^T V x)
            Vx = self.V[a] @ context
            var_term = context @ Vx
            if var_term < 0:
                var_term = 0.0
            bonus = c_val * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1

        # Track prediction error for adaptive gamma
        pred = self.theta[arm] @ context
        pred_err = abs(reward - pred)
        self.pred_error_window.append(pred_err)
        if len(self.pred_error_window) > self.pred_error_window_max:
            self.pred_error_window.pop(0)
        if len(self.pred_error_window) == 10 and self.base_pred_error is None:
            self.base_pred_error = max(np.mean(self.pred_error_window), 1e-8)

        # Track per-arm reward stats
        self.counts[arm] += 1
        self.arm_reward_sums[arm] += reward
        self.arm_reward_sq_sums[arm] += reward * reward

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance with forgetting + ridge
        # V_new = (gamma * V_old^{-1} + x x^T + lambda I)^{-1}
        # Using Sherman-Morrison on A = gamma * V^{-1} + lambda I
        # Actually, let's use the standard discounted update:
        # A_new = gamma * A_old + x x^T  (where A = V^{-1})
        # But we track V = A^{-1}.
        #
        # V_new = (gamma * V^{-1} + x x^T)^{-1}
        # = (1/gamma) * V - (1/gamma) * V x x^T V / (gamma + x^T V x)
        #
        # With ridge: add lambda I to the "A" side means V gets an extra term.
        # Simpler: incorporate ridge into the update by adding lambda * I to the
        # effective inverse, which means subtracting a small amount from V.
        #
        # Alternative: just use the standard update and periodically re-regularize.
        # For now, use the standard Sherman-Morrison update.

        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-10:
            V_new = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            V_new = np.eye(self.d) * (1.0 / self.lambda_ridge)

        # Symmetrize for numerical stability
        V_new = 0.5 * (V_new + V_new.T)

        # Clip: ensure positive semi-definite by adding small ridge if needed
        # Quick check: if any diagonal is too small, reset
        diag = np.diag(V_new)
        if np.any(diag < 1e-8):
            V_new = np.eye(self.d) * (1.0 / self.lambda_ridge)

        self.V[arm] = V_new

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Update theta
        self._update_theta(arm)
# EVOLVE-BLOCK-END
