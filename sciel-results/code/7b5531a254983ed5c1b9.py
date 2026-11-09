"""
Direct approach for Non-stationary Linear Contextual Bandit.
Uses RLS with adaptive forgetting factor (lambda) and adaptive exploration (alpha)
driven by a global drift metric based on prediction residuals.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Non-stationary LinUCB.
    
    Features:
    1. Recursive Least Squares (RLS) with per-arm forgetting factor.
    2. Global drift detection based on recent prediction errors using log-sum-exp.
    3. Adaptive exploration bonus that increases during high drift but decays over time.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        # Safety margin: Evaluator kills at 150s. We stop risky updates after 140s.
        self.deadline = self.start_time + 140.0 

        # Drift Detection State
        # We track the residual of the selected arm to detect global regime changes.
        # If the error spikes, we assume a shift.
        self.residual_hist = []
        self.max_hist_len = 20
        self.min_lambda = 0.92
        self.max_lambda = 0.995
        
        # Per-arm adaptive lambda state
        self.arm_lambda = [0.98 for _ in range(n_arms)]
        self.arm_residual_hist = [[] for _ in range(n_arms)]
        self.arm_hist_len = 10
        
        # Exploration Parameters
        self.base_alpha = 1.0
        self.alpha_drift_boost = 1.2
        
        # Counters
        self.counts = [0] * n_arms
        
        # Residual variance estimate for adaptive scaling
        self.residual_var = 1.0
        
        # Smoothed drift metric (exponential moving average)
        self.smoothed_drift = 0.0
        self.drift_ema_alpha = 0.3

    def _compute_drift_metric(self):
        """
        Compute a drift metric based on recent residuals using smoothed log-sum-exp.
        Uses adaptive variance scaling for better non-stationarity capture.
        Returns smoothed drift via exponential moving average.
        """
        if len(self.residual_hist) < 2:
            return self.smoothed_drift
        
        recent = np.array(self.residual_hist[-self.max_hist_len:])
        
        # Adaptive variance scaling: estimate typical residual scale
        # Use median absolute deviation as robust scale estimator
        med = np.median(recent)
        mad = np.median(np.abs(recent - med)) + 1e-8
        scaled_residuals = (recent - med) / mad
        
        # Smoothed log-sum-exp based error metric
        # This provides a smooth surrogate for max-based drift detection
        scale_param = 0.5  # Temperature for log-sum-exp
        shifted = scaled_residuals / scale_param
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        
        # Also compute mean absolute scaled error
        mean_abs_scaled = np.mean(np.abs(scaled_residuals))
        
        # Combine: lse captures outliers, mean_abs captures systematic shift
        drift_signal = lse + mean_abs_scaled
        
        # Normalize to [0, 1] with smooth sigmoid-like mapping
        # Map [1.0, 4.0] to [0, 1] smoothly
        t = (drift_signal - 1.0) / 3.0
        # Smooth step function: 3t^2 - 2t^3
        t_clamped = np.clip(t, 0, 1)
        raw_drift = 3 * t_clamped**2 - 2 * t_clamped**3
        
        # Exponential moving average for smoothing
        self.smoothed_drift = (1 - self.drift_ema_alpha) * self.smoothed_drift + self.drift_ema_alpha * raw_drift
        
        return float(min(1.0, max(0.0, self.smoothed_drift)))

    def _compute_arm_drift(self, arm):
        """
        Compute per-arm drift metric from that arm's recent residuals.
        """
        hist = self.arm_residual_hist[arm]
        if len(hist) < 2:
            return 0.0
        
        recent = np.array(hist[-self.arm_hist_len:])
        med = np.median(recent)
        mad = np.median(np.abs(recent - med)) + 1e-8
        scaled = (recent - med) / mad
        
        # Simple mean absolute scaled error
        mean_abs = np.mean(np.abs(scaled))
        
        # Map to [0,1]
        t = np.clip((mean_abs - 0.5) / 2.0, 0, 1)
        return float(3 * t**2 - 2 * t**3)

    def select(self, context):
        # Deadline Guard
        if time.time() > self.deadline:
            # Fallback: Greedy based on current theta
            context = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Compute Global Drift Metric (smoothed)
        drift_metric = self._compute_drift_metric()
        
        # Adapt Alpha (Exploration Bonus) with temporal decay
        t = self.t
        # Smooth decay: exploration decreases over time
        decay_factor = 1.0 / (1.0 + 0.008 * t)
        
        # Base alpha decays
        base_alpha_decayed = self.base_alpha * decay_factor
        
        # Drift boost: higher during high drift
        drift_boost = self.alpha_drift_boost * (1.0 / (1.0 + 0.003 * t))
        
        alpha = base_alpha_decayed + drift_boost * drift_metric
        
        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            try:
                # Solve for theta
                theta = np.linalg.solve(A, b)
                mu = theta @ context
                
                # Compute Variance: x^T A^{-1} x
                v_vec = np.linalg.solve(A, context)
                variance = context @ v_vec
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            # Per-arm exploration bonus adjustment
            arm_drift = self._compute_arm_drift(a)
            arm_alpha = alpha * (1.0 + 0.5 * arm_drift)
            
            ucb_score = mu + arm_alpha * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update: If we are close to limit, skip complex updates
        if time.time() > self.deadline:
             return

        # 1. Compute Prediction Error for Drift Detection
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
            mu_pred = theta @ context
        except np.linalg.LinAlgError:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        abs_residual = abs(residual)
        
        # Global residual history
        self.residual_hist.append(abs_residual)
        if len(self.residual_hist) > self.max_hist_len * 2:
            self.residual_hist.pop(0)
        
        # Per-arm residual history
        self.arm_residual_hist[arm].append(abs_residual)
        if len(self.arm_residual_hist[arm]) > self.arm_hist_len * 2:
            self.arm_residual_hist[arm].pop(0)
        
        # 2. Update Adaptive Parameters
        drift_metric = self._compute_drift_metric()
        
        # Per-arm adaptive lambda
        arm_drift = self._compute_arm_drift(arm)
        # Combine global and per-arm drift for this arm's lambda
        combined_drift = 0.6 * drift_metric + 0.4 * arm_drift
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * combined_drift
        
        # Smoothly adapt per-arm lambda
        self.arm_lambda[arm] = 0.7 * self.arm_lambda[arm] + 0.3 * lam
        lam = self.arm_lambda[arm]
        
        # 3. Update RLS State
        # A <- lam * A + x x^T
        # b <- lam * b + r x
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        self.counts[arm] += 1

# EVOLVE-BLOCK-END
