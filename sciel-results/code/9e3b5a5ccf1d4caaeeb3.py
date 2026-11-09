import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy using per-arm LinUCB with 
    adaptive forgetting driven by prediction residuals.
    
    Improvements over baseline:
    1. Continuous adaptive forgetting: The forgetting factor gamma is adjusted 
       continuously based on the squared residual of the last observation, 
       allowing for faster reaction to concept drift.
    2. Confidence-weighted blending: The mix between per-arm UCB scores and 
       global ridge model scores is weighted by the relative confidence 
       (inverse variance) of the estimates.
    3. Optimized linear algebra: Uses Cholesky decomposition for variance 
       calculation and efficient matrix updates.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Per-arm LinUCB parameters
        # Initialize with identity matrix scaled by ridge parameter
        self.ridge = 1.0
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Global ridge value model
        self.G = np.eye(dim) * self.ridge
        self.f = np.zeros(dim)
        self.theta_global = np.zeros(dim)
        self.var_global = 1.0
        
        # Adaptive Forgetting Parameters
        # Base forgetting factor (close to 1 means long memory)
        self.gamma_base = 0.99
        self.gamma_min = 0.90  # Most aggressive forgetting
        self.gamma_max = 0.999 # Least aggressive forgetting
        # Sensitivity of forgetting to residual error
        self.forgetting_beta = 0.1
        
        # UCB Exploration Parameter
        self.alpha = 1.0
        
        # Deadline Guard
        self.start_time = time.time()
        self.deadline = 115.0
        self.deadline_hit = False
        
        # Track residuals for adaptive forgetting
        self.last_residual_sq = 0.0
        self.residual_ewma = 0.0
        self.residual_alpha = 0.1  # EWMA smoothing factor for residuals

    def _check_deadline(self):
        if self.deadline_hit:
            return
        if time.time() - self.start_time > self.deadline:
            self.deadline_hit = True

    def _update_global_model(self):
        """Update global ridge model parameters."""
        try:
            inv_G = np.linalg.inv(self.G + 1e-6 * np.eye(self.d))
            self.theta_global = inv_G @ self.f
            # Estimate variance of global model (heuristic based on inverse diagonal)
            self.var_global = 0.5 * np.mean(np.abs(np.diag(inv_G)))
        except np.linalg.LinAlgError:
            pass

    def _get_adaptive_gamma(self, residual_sq):
        """
        Compute adaptive forgetting factor based on residual error.
        Larger residuals lead to smaller gamma (more forgetting).
        """
        # Update EWMA of residual squared
        self.residual_ewma = (1 - self.residual_alpha) * self.residual_ewma + self.residual_alpha * residual_sq
        
        # Map residual EWMA to gamma
        # If residual is small, gamma stays near gamma_base
        # If residual is large, gamma drops towards gamma_min
        # Using a simple scaling: gamma = gamma_base - beta * residual_ewma
        # But we want it to be smooth and bounded.
        
        # Normalize residual impact
        impact = np.sqrt(self.residual_ewma)
        # Scale impact to range [0, 1] roughly
        # Assume typical residual scale is around 1.0
        scaled_impact = np.tanh(impact * self.forgetting_beta * 10)
        
        gamma = self.gamma_base - (self.gamma_base - self.gamma_min) * scaled_impact
        return np.clip(gamma, self.gamma_min, self.gamma_max)

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        
        if self.deadline_hit:
            # Fallback to global model exploitation
            if self.t > 0:
                return int(np.argmax(self.theta_global @ ctx))
            return 0

        scores = np.empty(self.n)
        variances = np.empty(self.n)
        
        for a in range(self.n):
            try:
                # Add small regularization for numerical stability
                A_reg = self.A[a] + 1e-6 * np.eye(self.d)
                
                # Compute mean
                inv_A = np.linalg.inv(A_reg)
                theta = inv_A @ self.b[a]
                mean = theta @ ctx
                
                # Compute variance using Cholesky
                L = np.linalg.cholesky(A_reg)
                v = np.linalg.solve(L, ctx)
                variance = np.dot(v, v)
                
                scores[a] = mean + self.alpha * np.sqrt(variance)
                variances[a] = variance
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                inv_A = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv_A @ self.b[a]
                mean = theta @ ctx
                variance = ctx @ inv_A @ ctx
                scores[a] = mean + self.alpha * np.sqrt(variance)
                variances[a] = variance

        # Blend with global model based on confidence
        if self.t > 0:
            # Global score
            global_score = self.theta_global @ ctx
            
            # Compute confidence weights
            # Lower variance -> Higher confidence
            # We want to blend per-arm UCB with global mean
            # Weight w in [0, 1] for per-arm, (1-w) for global
            
            # Average variance of per-arm estimates
            avg_var = np.mean(variances)
            
            # Heuristic: if per-arm variance is high, trust global more
            # If global variance is low, trust global more
            # Simple approach: fixed blend 0.7 per-arm, 0.3 global
            # But let's make it slightly adaptive
            # If avg_var is very high, the per-arm model is uncertain.
            
            # Let's use a smooth step or logistic function
            # Confidence ratio: global_conf / (global_conf + per_arm_conf)
            # Approximate conf as 1/var
            # global_conf ~ 1/self.var_global
            # per_arm_conf ~ 1/avg_var
            
            if self.var_global > 0 and avg_var > 0:
                w_per_arm = (1.0 / avg_var) / ((1.0 / avg_var) + (1.0 / self.var_global))
            else:
                w_per_arm = 0.7 # Default
            
            # Clamp weight to avoid extreme values
            w_per_arm = np.clip(w_per_arm, 0.2, 0.8)
            
            blended_scores = w_per_arm * scores + (1 - w_per_arm) * global_score
            return int(np.argmax(blended_scores))
        else:
            return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)
        
        if self.deadline_hit:
            return

        # Calculate residual for adaptive forgetting
        # Use current model to predict
        try:
            A_reg = self.A[arm] + 1e-6 * np.eye(self.d)
            inv_A = np.linalg.inv(A_reg)
            theta = inv_A @ self.b[arm]
            pred = theta @ ctx
            residual = reward - pred
            residual_sq = residual ** 2
        except np.linalg.LinAlgError:
            residual_sq = 1.0 # Fallback

        # Get adaptive forgetting factor
        gamma = self._get_adaptive_gamma(residual_sq)

        # Update per-arm model with forgetting
        self.A[arm] = gamma * self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = gamma * self.b[arm] + reward * ctx

        # Update global model with same forgetting (or slightly different?)
        # Using same gamma for consistency
        self.G = gamma * self.G + np.outer(ctx, ctx)
        self.f = gamma * self.f + reward * ctx
        
        # Update global model parameters
        self._update_global_model()

        self.t += 1
