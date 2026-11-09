"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary Linear Contextual Bandit with Multi-Timescale Adaptive Forgetting.
    
    Maintains two parallel LinUCB estimators per arm:
    - Short-window (fast lambda): rapid detection of recent weight shifts
    - Long-window (slow lambda): stable baseline estimation
    
    Arm selection uses the short-window UCB to prioritize immediate adaptation.
    The long-window model provides regularization and confidence calibration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Regularization
        self.reg = 0.1
        
        # Short-window models (fast adaptation)
        self.A_short = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.lam_short = [0.92 for _ in range(n_arms)]
        self.lam_short_min = 0.82
        self.lam_short_max = 0.96
        
        # Long-window models (stable baseline)
        self.A_long = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.lam_long = [0.985 for _ in range(n_arms)]
        self.lam_long_min = 0.95
        self.lam_long_max = 0.995
        
        # Adaptive lambda tracking
        self.residuals_short = [deque(maxlen=15) for _ in range(n_arms)]
        self.residuals_long = [deque(maxlen=25) for _ in range(n_arms)]
        self.lambda_adapt_rate = 0.10
        self.residual_mad = [1.0 for _ in range(n_arms)]
        self.residual_mad_long = [1.0 for _ in range(n_arms)]
        
        # Global drift tracking
        self.recent_rewards = deque(maxlen=20)
        self.global_reward_avg = 0.0
        self.global_reward_var = 1.0
        
        # Exploration coefficient
        self.alpha = 1.5
        self.alpha_init = 1.5
        self.alpha_min = 0.6
        self.alpha_decay_rate = 0.00015
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0

    def _solve(self, A, b):
        """Solve A x = b with fallback for singular matrices."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_lambda_short(self, arm, residual_sq):
        """Adapt short-window forgetting factor using MAD-based robust scaling."""
        self.residuals_short[arm].append(residual_sq)
        
        if len(self.residuals_short[arm]) < 4:
            return

        recent_res = np.array(list(self.residuals_short[arm]))
        median_res = np.median(recent_res)
        mad = np.median(np.abs(recent_res - median_res)) + 1e-6
        
        # Smoothly update the running MAD estimate
        self.residual_mad[arm] = 0.8 * self.residual_mad[arm] + 0.2 * mad
        
        # Robust error metric: normalize by MAD to get a scale-free signal
        error_metric = median_res / (self.residual_mad[arm] + 1e-6)
        
        # Larger errors -> smaller lambda (faster forgetting)
        sens = 0.05
        adjustment = -sens * error_metric
        
        new_lambda = self.lam_short[arm] + self.lambda_adapt_rate * adjustment
        self.lam_short[arm] = max(self.lam_short_min, min(self.lam_short_max, new_lambda))

    def _update_lambda_long(self, arm, residual_sq):
        """Adapt long-window forgetting factor using MAD-based robust scaling (slower)."""
        self.residuals_long[arm].append(residual_sq)
        
        if len(self.residuals_long[arm]) < 6:
            return

        recent_res = np.array(list(self.residuals_long[arm]))
        median_res = np.median(recent_res)
        mad = np.median(np.abs(recent_res - median_res)) + 1e-6
        
        # Smoothly update the running MAD estimate
        self.residual_mad_long[arm] = 0.85 * self.residual_mad_long[arm] + 0.15 * mad
        
        # Robust error metric
        error_metric = median_res / (self.residual_mad_long[arm] + 1e-6)
        
        sens = 0.025
        adjustment = -sens * error_metric
        
        new_lambda = self.lam_long[arm] + 0.04 * adjustment
        self.lam_long[arm] = max(self.lam_long_min, min(self.lam_long_max, new_lambda))

    def _check_global_drift(self, reward):
        """
        Detect global drift using CUSUM-style cumulative sum for more sensitivity
        to sustained shifts rather than single-point spikes.
        """
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) < 8:
            return False
            
        recent = np.array(list(self.recent_rewards))
        current_mean = np.mean(recent)
        current_var = np.var(recent)
        
        # Update running stats
        self.global_reward_avg = 0.92 * self.global_reward_avg + 0.08 * current_mean
        self.global_reward_var = 0.88 * self.global_reward_var + 0.12 * current_var
        
        # CUSUM-like: compare recent half to earlier half
        half = len(recent) // 2
        early_mean = np.mean(recent[:half])
        late_mean = np.mean(recent[half:])
        diff = late_mean - early_mean
        
        scale = max(0.5, np.sqrt(self.global_reward_var + 1e-6))
        drift_threshold = 0.4 * scale
        
        if abs(diff) > drift_threshold:
            return True
            
        return False

    def _select_arm(self, context):
        """Select arm using short-window UCB with long-window confidence blending."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        total_obs = sum(self.counts)
        # Smoother alpha decay: use sqrt scaling for gentler exploration reduction
        self.alpha = max(self.alpha_min, self.alpha_init * (1.0 - 0.4 * np.log1p(total_obs) / np.log1p(5000)))

        scores = []
        for a in range(self.n):
            # Short-window estimate (primary for adaptation)
            A_s = self.A_short[a]
            b_s = self.b_short[a]
            theta_s = self._solve(A_s, b_s)
            mu_s = theta_s @ context
            
            try:
                A_inv_s_ctx = np.linalg.solve(A_s, context)
                v_s = context @ A_inv_s_ctx
            except np.linalg.LinAlgError:
                v_s = 1.0
            
            # Long-window estimate (for regularization/confidence)
            A_l = self.A_long[a]
            b_l = self.b_long[a]
            theta_l = self._solve(A_l, b_l)
            mu_l = theta_l @ context
            
            try:
                A_inv_l_ctx = np.linalg.solve(A_l, context)
                v_l = context @ A_inv_l_ctx
            except np.linalg.LinAlgError:
                v_l = 1.0
            
            # Blend: primary signal from short-window, regularize with long-window
            # Use confidence-based weighting: higher confidence in short-window
            # when it has more data relative to long-window
            conf_s = 1.0 / (1.0 + max(v_s, 1e-6))
            conf_l = 1.0 / (1.0 + max(v_l, 1e-6))
            
            # Dynamic blend: weight short-window more when its confidence is high
            w_s = 0.65 + 0.35 * conf_s
            w_l = 1.0 - w_s
            mu_blend = w_s * mu_s + w_l * mu_l
            
            # Blend variance: short-window variance dominates for exploration
            v_blend = 0.75 * v_s + 0.25 * v_l
            
            bonus = self.alpha * np.sqrt(max(v_blend, 0.0))
            
            scores.append(mu_blend + bonus)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return arm with highest historical average reward if available
            # Since we don't track averages explicitly here, we can just return 0
            # or a random arm. To be safe and deterministic, return 0.
            # A better fallback would be tracking moving averages, but for simplicity and speed:
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction errors for both windows
        A_s = self.A_short[arm]
        b_s = self.b_short[arm]
        theta_s = self._solve(A_s, b_s)
        mu_pred_s = theta_s @ context
        residual_s = reward - mu_pred_s
        residual_sq_s = residual_s ** 2
        
        A_l = self.A_long[arm]
        b_l = self.b_long[arm]
        theta_l = self._solve(A_l, b_l)
        mu_pred_l = theta_l @ context
        residual_l = reward - mu_pred_l
        residual_sq_l = residual_l ** 2

        # Update adaptive lambdas
        self._update_lambda_short(arm, residual_sq_s)
        self._update_lambda_long(arm, residual_sq_l)

        # Check for global drift
        is_global_drift = self._check_global_drift(reward)
        if is_global_drift:
            for a in range(self.n):
                self.lam_short[a] = max(self.lam_short_min, self.lam_short[a] * 0.90)
                self.lam_long[a] = max(self.lam_long_min, self.lam_long[a] * 0.95)
                # Also reset MAD estimates to adapt to new scale
                self.residual_mad[a] = max(0.1, self.residual_mad[a] * 0.7)
                self.residual_mad_long[a] = max(0.1, self.residual_mad_long[a] * 0.7)

        # Update short-window model
        lam_s = self.lam_short[arm]
        self.A_short[arm] = lam_s * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = lam_s * self.b_short[arm] + reward * context
        
        # Update long-window model
        lam_l = self.lam_long[arm]
        self.A_long[arm] = lam_l * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_l * self.b_long[arm] + reward * context
        
        # Update count
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
