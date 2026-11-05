"""Non-stationary linear contextual bandit with Huber-loss RLS and adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using per-arm Robust RLS with:
    - Huber-loss-based update to mitigate sensitivity to outliers/regime shifts
    - Adaptive forgetting factor driven by innovation (prediction residual)
    - Sliding-window inverse covariance reset when innovation exceeds dynamic threshold
    - UCB exploration with adaptive coefficient
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s kill

        # Per-arm state
        # theta: weight estimate (dim,)
        # P: inverse covariance (dim, dim) -- actually V = A^{-1} in RLS
        # We store V (inverse of the covariance matrix of the estimator)
        # In RLS: V_new = (V_old / gamma + x x^T)^{-1}
        # But for numerical stability and to match Kalman-style, let's use P = covariance
        # Actually, the current program uses V = A^{-1}. Let's stick with that convention
        # V is the inverse of the accumulated (discounted) Hessian.
        # theta = V @ b

        # Initialize with large uncertainty (small V, or rather V ~ I/lambda)
        # If V = A^{-1}, large A means small V. We want large initial uncertainty, so small A, large V.
        # Let's initialize V = I * 10.0 (large uncertainty)
        self.V = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.base_gamma = 0.97  # Base forgetting factor
        self.gamma_min = 0.85
        self.gamma_max = 0.995
        self.C = 1.2  # Exploration coefficient
        self.C_min = 0.2
        self.C_max = 5.0

        # Huber loss parameters
        self.huber_delta = 1.5  # Delta for Huber loss
        
        # Innovation tracking for adaptive gamma and reset
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.3
        self.reset_threshold = 3.0  # Dynamic threshold multiplier for sqrt(Pxx)

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _huber_weight(self, residual, sigma):
        """
        Compute Huber loss weight for the residual.
        If |residual| < delta * sigma, weight = 1 (Gaussian-like)
        If |residual| >= delta * sigma, weight = (delta * sigma) / |residual|
        This effectively caps the gradient contribution of large outliers.
        """
        abs_res = abs(residual)
        threshold = self.huber_delta * sigma
        if abs_res <= threshold:
            return 1.0
        else:
            return threshold / abs_res

    def _adaptive_gamma(self, arm, innov_norm):
        """
        Adapt forgetting factor based on normalized innovation.
        High innovation -> lower gamma (faster forgetting)
        """
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # innov_norm is |r - pred| / sqrt(Pxx + R)
        # Use EMA to smooth
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov_norm ** 2)
        
        ema = self.innov_ema[arm]
        
        # Map EMA to gamma
        # If EMA is high (large errors), decrease gamma
        # gamma = base - k * (EMA - 1) if EMA > 1
        # Let's use a smooth function
        excess = max(0.0, ema - 1.0)
        gamma = self.base_gamma - 0.05 * excess
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 1.5
        # Decay over time but keep some exploration
        time_decay = max(0.5, 1.0 - 0.05 * (self.t / 1000.0))
        return self.C * time_decay

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
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
            # Uncertainty is sqrt(x^T V x) since V = A^{-1} is the covariance of theta
            bonus = c_val * np.sqrt(max(context @ self.V[a] @ context, 0.0))
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1

        self.counts[arm] += 1

        # Get current estimates
        V = self.V[arm]
        b = self.b[arm]
        theta = V @ b
        pred = theta @ context
        innov = reward - pred

        # Estimate observation noise variance (approx)
        # Pxx = x^T V x is the variance of the prediction
        Pxx = float(context @ V @ context)
        # Assume observation noise R is small, or fold it in
        R = 0.1
        sigma = np.sqrt(Pxx + R)
        
        # Normalized innovation
        if sigma > 1e-9:
            innov_norm = abs(innov) / sigma
        else:
            innov_norm = 0.0

        # Check for reset condition: if innovation is very large, reset V
        # Threshold: 3 * sigma (3-sigma rule)
        if innov_norm > self.reset_threshold and self.counts[arm] > 10:
            # Reset V to a moderate uncertainty level
            self.V[arm] = np.eye(self.d) * 5.0
            # Keep b? Or reset b?
            # If we reset V, theta = V @ b. If we keep b, theta changes.
            # Better to reset both or just V and re-estimate.
            # Let's just reset V to encourage exploration of this arm's weights
            # But we should probably also dampen b or reset it?
            # Resetting V alone effectively "forgets" the past data's influence on the precision.
            # Let's try resetting V only first.
            pass

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm, innov_norm)

        # Huber weight
        w = self._huber_weight(innov, sigma)

        # RLS Update with Forgetting and Huber
        # V_new = (V_old / gamma + w * x x^T)^{-1}
        # Using Sherman-Morrison:
        # Let A = V_old / gamma
        # V_new = (A + w x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1} w) / (1 + w x^T A^{-1} x)
        # A^{-1} = gamma * V_old
        # So V_new = gamma * V_old - (gamma * V_old x x^T V_old w) / (1 + w x^T V_old x)
        
        Vx = V @ context
        xVx = context @ Vx
        denom = 1.0 + w * xVx

        if denom > 1e-12:
            # V_new = gamma * V - (gamma * w / denom) * outer(Vx, Vx)
            self.V[arm] = gamma * V - (gamma * w / denom) * np.outer(Vx, Vx)
        else:
            # Fallback
            self.V[arm] = np.eye(self.d) * 5.0

        # Update b
        # b_new = gamma * b + w * reward * x
        self.b[arm] = gamma * b + w * reward * context

# EVOLVE-BLOCK-END
