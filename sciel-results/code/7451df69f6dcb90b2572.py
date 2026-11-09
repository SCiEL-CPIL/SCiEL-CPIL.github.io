"""
Combined approach for Non-stationary Linear Contextual Bandit.
Uses Per-Arm RLS with adaptive forgetting factor and exploration,
driven by per-arm smoothed log-sum-exp drift metrics.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive Non-stationary LinUCB with Per-Arm Drift Detection.
    
    Features:
    1. Recursive Least Squares (RLS) maintaining inverse covariance P directly.
    2. Per-arm drift detection using smoothed log-sum-exp on residuals.
    3. Per-arm adaptive forgetting factor (lambda) and exploration bonus (alpha).
    4. Temporal decay of exploration to reduce over-exploration in later rounds.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State: P is inverse of A (covariance), b is sum of r*x
        # Initialize P as identity (A = I)
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        # Safety margin: Evaluator kills at 150s. We stop risky updates after 140s.
        self.deadline = self.start_time + 140.0 

        # Drift Detection State per Arm
        # Store residuals for each arm
        self.residual_hists = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Adaptive Parameters per Arm
        self.lambdas = [0.98] * n_arms
        self.min_lambda = 0.90
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.5
        
        # Counters
        self.counts = [0] * n_arms
        
        # Residual variance estimates per arm for adaptive scaling
        self.residual_vars = [1.0] * n_arms

    def _compute_arm_drift(self, arm_idx):
        """
        Compute a drift metric for a specific arm based on its recent residuals.
        Uses smoothed log-sum-exp for a smooth surrogate of max-based detection.
        """
        hist = self.residual_hists[arm_idx]
        if len(hist) < 2:
            return 0.0
        
        recent = np.array(list(hist))
        
        # Adaptive variance scaling: estimate typical residual scale
        med = np.median(recent)
        mad = np.median(np.abs(recent - med)) + 1e-8
        scaled_residuals = (recent - med) / mad
        
        # Smoothed log-sum-exp based error metric
        scale_param = 0.5  # Temperature for log-sum-exp
        shifted = scaled_residuals / scale_param
        max_shifted = np.max(shifted)
        # Numerical stability: subtract max before exp
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        
        # Also compute mean absolute scaled error
        mean_abs_scaled = np.mean(np.abs(scaled_residuals))
        
        # Combine: lse captures outliers, mean_abs captures systematic shift
        drift_signal = lse + mean_abs_scaled
        
        # Normalize to [0, 1] with smooth sigmoid-like mapping
        # Typical values: lse ~ 1-3, mean_abs ~ 0.5-1.5
        # So drift_signal typically ranges 1.5-4.5
        # Map [1.0, 4.0] to [0, 1] smoothly
        t = (drift_signal - 1.0) / 3.0
        t_clamped = np.clip(t, 0, 1)
        drift_metric = 3 * t_clamped**2 - 2 * t_clamped**3
        
        return float(min(1.0, max(0.0, drift_metric)))

    def select(self, context):
        # Deadline Guard
        if time.time() > self.deadline:
            # Fallback: Greedy based on current theta
            context = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                try:
                    # theta = P * b
                    theta = self.P[a] @ self.b[a]
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Temporal decay for exploration
        # Decay rate: after ~100 rounds, exploration bonus is reduced by ~50%
        decay_factor = 1.0 / (1.0 + 0.01 * self.t)
        
        scores = []
        for a in range(self.n):
            P = self.P[a]
            b = self.b[a]
            
            try:
                # theta = P * b
                theta = P @ b
                mu = theta @ context
                
                # Compute Variance: x^T P x
                # P is the inverse covariance, so x^T P x is the variance term
                variance = context @ (P @ context)
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            # Compute Per-Arm Drift Metric
            drift_metric = self._compute_arm_drift(a)
            
            # Adapt Alpha (Exploration Bonus)
            # Base alpha decays over time
            base_alpha_decayed = self.base_alpha * decay_factor
            # Drift boost also decays but less aggressively
            drift_boost_decayed = self.alpha_drift_boost * (1.0 / (1.0 + 0.005 * self.t))
            
            alpha = base_alpha_decayed + drift_boost_decayed * drift_metric
            
            ucb_score = mu + alpha * np.sqrt(max(variance, 0.0))
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
        P = self.P[arm]
        b = self.b[arm]
        try:
            theta = P @ b
            mu_pred = theta @ context
        except np.linalg.LinAlgError:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        self.residual_hists[arm].append(abs(residual))
        
        # 2. Update Residual Variance Estimate
        if len(self.residual_hists[arm]) >= 5:
            recent_res = np.array(list(self.residual_hists[arm][-min(10, len(self.residual_hists[arm])):]))
            self.residual_vars[arm] = float(np.var(recent_res)) + 1e-8
            
        # 3. Compute Per-Arm Drift Metric
        drift_metric = self._compute_arm_drift(arm)
        
        # 4. Adapt Lambda (Forgetting Factor)
        # High drift -> Lower Lambda (Forget Faster)
        # Linear interpolation
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * drift_metric
        self.lambdas[arm] = lam
        
        # 5. Update RLS State using Matrix Inversion Lemma
        # P_new = (1/lam) * (P_old - (P_old x x^T P_old) / (lam + x^T P_old x))
        # b_new = (1/lam) * (b_old + reward * x)
        
        Px = P @ context
        xPx = context @ Px
        denom = lam + xPx
        
        if denom < 1e-10:
            # Numerical stability fallback: ignore update or reset
            return
            
        # Update P
        # P <- (1/lam) * (P - (P x x^T P) / denom)
        # Note: (P x) is Px, so (P x x^T P) = Px * (x^T P) = Px * (Px^T)
        # So P update is: P - (Px * Px^T) / denom, then scale by 1/lam
        P_update = P - np.outer(Px, Px) / denom
        self.P[arm] = (1.0 / lam) * P_update
        
        # Update b
        self.b[arm] = (1.0 / lam) * (b + reward * context)
        
        self.counts[arm] += 1

# EVOLVE-BLOCK-END
