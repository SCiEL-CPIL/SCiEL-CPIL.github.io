"""
Refined Non-stationary Linear Contextual Bandit.
Improvements over the seed:
1. Per-arm drift detection using smoothed log-sum-exp (soft-max) on residuals.
2. Adaptive forgetting factor (lambda) per arm, reacting to local instability.
3. Refined exploration bonus scaling with drift and temporal decay.
4. Maintains the RLS structure for efficiency.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Non-stationary LinUCB with Per-Arm Drift Detection.
    
    Features:
    1. Recursive Least Squares (RLS) with per-arm adaptive forgetting factor.
    2. Per-arm drift detection using a smooth soft-max (log-sum-exp) of recent residuals.
    3. Adaptive exploration bonus that increases during high drift and decays over time.
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

        # Drift Detection State (Per Arm)
        self.residual_hists = [[] for _ in range(n_arms)]
        self.max_hist_len = 15  # Slightly shorter window for faster reaction
        
        # Adaptive Forgetting Factors
        self.min_lambda = 0.85  # Faster forgetting when drift is high
        self.max_lambda = 0.99  # Slower forgetting when stable
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.8  # Stronger boost during drift
        
        # Counters
        self.counts = [0] * n_arms

    def _compute_arm_drift(self, arm_idx):
        """
        Compute a drift metric for a specific arm using smooth log-sum-exp.
        Returns a value in [0, 1].
        """
        hist = self.residual_hists[arm_idx]
        if len(hist) < 3:
            return 0.0
        
        recent = np.array(hist[-self.max_hist_len:])
        
        # Normalize residuals by their median absolute deviation (robust scale)
        med = np.median(recent)
        mad = np.median(np.abs(recent - med)) + 1e-8
        scaled = (recent - med) / mad
        
        # Soft-max (log-sum-exp) to capture the "worst" error smoothly
        # Temperature parameter controls the sharpness of the max approximation
        temp = 0.5
        shifted = scaled / temp
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        
        # The lse value is roughly proportional to the max error + log(N)
        # We want to map this to [0, 1].
        # Typical scaled errors are in [-2, 2] range. Max is ~2.
        # lse for max=2, N=15 is approx 2 + log(15) ~ 2 + 2.7 = 4.7 (if all were max, which they aren't)
        # Actually, lse approximates max(x) + log(sum(exp(x-max)))
        # If one outlier is 3 and others are 0: lse ~ 3 + log(1+14) ~ 3+2.7 = 5.7
        # If all are 0: lse ~ 0 + log(15) ~ 2.7
        
        # Let's normalize based on the scale of 'scaled'.
        # A drift signal of 0 should correspond to lse ~ log(N) (no error).
        # A drift signal of 1 should correspond to a significant error, e.g., scaled error of 2.0.
        # Target lse for drift=1: 2.0 + log(15) ~ 4.7? 
        # Let's use a simple linear map from [log(N), log(N) + 2.5] to [0, 1]
        # This assumes an error of 2.5 MADs is "full drift".
        
        baseline_lse = np.log(len(recent))
        target_lse = baseline_lse + 2.5
        
        drift_raw = (lse - baseline_lse) / (target_lse - baseline_lse)
        
        # Clamp and smooth step
        t = np.clip(drift_raw, 0, 1)
        drift_metric = 3 * t**2 - 2 * t**3
        
        return float(min(1.0, max(0.0, drift_metric)))

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
        
        # Compute Global Drift Metric (max of arm drifts) for exploration boost
        # We use the max drift across arms to determine if the *environment* is changing
        arm_drifts = [self._compute_arm_drift(a) for a in range(self.n)]
        global_drift = max(arm_drifts) if arm_drifts else 0.0
        
        # Adapt Alpha (Exploration Bonus) with temporal decay
        t = self.t
        # Decay rate: exploration bonus reduces over time to favor exploitation
        # But we keep a floor to avoid zero exploration
        decay_factor = 1.0 / (1.0 + 0.008 * t)
        
        base_alpha_decayed = self.base_alpha * decay_factor
        drift_boost_decayed = self.alpha_drift_boost * (1.0 / (1.0 + 0.004 * t))
        
        alpha = base_alpha_decayed + drift_boost_decayed * global_drift
        
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

            ucb_score = mu + alpha * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update
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
        self.residual_hists[arm].append(abs(residual))
        if len(self.residual_hists[arm]) > self.max_hist_len * 2:
            self.residual_hists[arm].pop(0)
        
        # 2. Compute Per-Arm Drift and Adapt Lambda
        drift_metric = self._compute_arm_drift(arm)
        
        # High drift -> Lower Lambda (Forget Faster)
        # Linear interpolation
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * drift_metric
        
        # 3. Update RLS State
        # A <- lam * A + x x^T
        # b <- lam * b + r x
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        self.counts[arm] += 1

# EVOLVE-BLOCK-END
