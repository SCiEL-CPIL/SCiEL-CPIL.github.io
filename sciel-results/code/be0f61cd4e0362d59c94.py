"""Non-stationary linear contextual bandit with robust RLS and sliding-window reset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using per-arm Robust RLS:
    - State: theta_a (dim-vector) for each arm
    - Observation model: r = x^T theta_a + noise
    - Huber-loss-based update for robustness to outliers
    - Sliding-window inverse covariance reset on large innovations
    - Adaptive process noise via innovation EMA
    - UCB exploration with count-dampened bonus
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s evaluator limit

        # RLS state per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Inverse covariance (P) - start with high uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Observation noise variance
        self.R = 0.1

        # Huber loss parameters
        self.huber_delta = 1.0  # threshold for Huber loss

        # Adaptive forgetting factor
        self.base_forgetting = 0.98  # base forgetting factor (slight drift tracking)
        self.forgetting_max = 0.90   # max forgetting (rapid drift)
        self.forgetting_min = 0.995  # min forgetting (stable regime)
        self.innov_threshold = 2.0   # threshold for innovation to trigger forgetting increase
        self.forgetting_alpha = 0.3  # adaptation rate for forgetting

        # EMA of squared normalized innovation per arm
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.3

        # Sliding window reset mechanism
        self.reset_threshold = 4.0  # innovation norm threshold for reset
        self.reset_decay = 0.95     # decay rate for P after reset trigger

        # UCB exploration
        self.C = 1.5
        self.C_min = 0.3
        self.C_max = 5.0
        self.count_scale = 0.4
        self.warmup_steps = 40

        # Time tracking
        self.t = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _huber_loss(self, residual):
        """Compute Huber loss gradient and effective variance."""
        abs_r = abs(residual)
        if abs_r <= self.huber_delta:
            # Quadratic region: gradient = residual, effective weight = 1
            grad = residual
            weight = 1.0
        else:
            # Linear region: gradient = sign(residual) * delta, effective weight = delta / |residual|
            grad = np.sign(residual) * self.huber_delta
            weight = self.huber_delta / abs_r
        return grad, weight

    def _adaptive_forgetting(self, arm, innov_norm_sq):
        """
        Compute adaptive forgetting factor based on innovation.
        High innovation -> lower forgetting (more weight on recent data)
        Low innovation -> higher forgetting (trust model more)
        """
        c = self.counts[arm]
        if c < 3:
            return self.base_forgetting

        # Update EMA of normalized innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * innov_norm_sq
        innov_ema = self.innov_ema[arm]

        # Map innovation EMA to forgetting factor
        # Low innovation (EMA ~ 1.0) -> high forgetting (0.995)
        # High innovation (EMA > threshold) -> low forgetting (0.90)
        excess = max(0.0, innov_ema - self.innov_threshold)
        # Exponential mapping: as excess grows, forgetting decreases
        forgetting = self.forgetting_max + (self.forgetting_min - self.forgetting_max) * np.exp(-0.5 * excess)
        # Clamp to valid range
        forgetting = np.clip(forgetting, self.forgetting_max, self.forgetting_min)
        return forgetting

    def _adaptive_C(self, arm):
        """Adapt exploration coefficient based on count and time."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 1.3
        # Gentle decay over time to maintain some exploration under non-stationarity
        time_decay = np.exp(-self.t / 2000.0)
        time_decay = max(0.4, time_decay)
        return self.C * time_decay

    def _should_reset(self, arm, innov_norm):
        """Check if innovation is large enough to trigger a reset."""
        return innov_norm > self.reset_threshold

    def _reset_covariance(self, arm):
        """Reset the inverse covariance to encourage re-exploration."""
        # Decay P towards initial high uncertainty
        self.P[arm] = self.reset_decay * self.P[arm] + (1.0 - self.reset_decay) * np.eye(self.d) * 10.0

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # UCB bonus: C * sqrt(x^T P x) / (1 + count_scale * sqrt(count))
            c_val = self._adaptive_C(a)
            std = np.sqrt(max(0.0, context @ self.P[a] @ context))
            c = self.counts[a]
            bonus = c_val * std / (1.0 + self.count_scale * np.sqrt(c))
            # Encourage exploring unseen arms early
            if c == 0:
                bonus += 1.0
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1
        self.counts[arm] += 1

        # RLS update for arm
        theta = self.theta[arm]
        P = self.P[arm]

        # Compute innovation (prediction residual)
        innov = reward - context @ theta
        S = context @ P @ context + self.R  # innovation variance
        if S < 1e-12:
            S = 1e-12
        innov_norm_sq = (innov * innov) / S  # normalized innovation squared
        innov_norm = np.sqrt(innov_norm_sq)

        # Check if reset is needed
        if self._should_reset(arm, innov_norm):
            self._reset_covariance(arm)
            P = self.P[arm]
            # Recompute S after reset
            S = context @ P @ context + self.R
            if S < 1e-12:
                S = 1e-12
            innov_norm_sq = (innov * innov) / S
            innov_norm = np.sqrt(innov_norm_sq)

        # Adaptive forgetting factor
        forgetting = self._adaptive_forgetting(arm, innov_norm_sq)

        # Huber loss for robustness
        grad, weight = self.huber_loss(innov)

        # RLS update with forgetting and Huber weight
        # P_new = (1/forgetting) * (P - P x (x^T P x + R/weight)^-1 x^T P)
        # theta_new = theta + P_new x * (grad / (x^T P_new x + R/weight))
        
        # Compute effective observation noise with Huber weight
        R_eff = self.R / weight if weight > 1e-10 else self.R

        # Inverse covariance update with forgetting
        P_forget = P / forgetting
        
        # Standard RLS gain
        P_x = P_forget @ context  # (dim,)
        denom = context @ P_x + R_eff
        if denom < 1e-12:
            denom = 1e-12
        K = P_x / denom  # (dim,)
        
        # Update theta
        self.theta[arm] = theta + K * grad
        
        # Update inverse covariance
        # P_new = P_forget - K x^T P_forget
        self.P[arm] = P_forget - np.outer(K, context) @ P_forget

        # Numerical stabilization
        if self.t % 100 == 0:
            for a in range(self.n):
                # Add small ridge to ensure positive definiteness
                self.P[a] += 1e-6 * np.eye(self.d)
                # Symmetrize
                self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
                # Clip diagonal to non-negative
                np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 0.0))
# EVOLVE-BLOCK-END
