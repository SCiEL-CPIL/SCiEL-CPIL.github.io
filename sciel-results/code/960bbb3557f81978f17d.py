"""
Non-stationary Linear Contextual Bandit Policy.

Uses an adaptive Linear UCB approach with:
1. Recursive Least Squares (RLS) for parameter estimation.
2. Dual time-scale adaptation: A fast "short-window" model for rapid drift tracking
   and a slow "long-window" model for stability.
3. Per-arm adaptive forgetting factors driven by local prediction errors.
4. "Shock" reset mechanism for arms that exhibit large prediction mismatches,
   allowing rapid recovery from regime changes.
5. Blended UCB selection that balances exploitation of the stable model with
   exploration driven by the adaptive model's uncertainty.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Long-window model parameters (stable baseline)
        self.A_long = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        
        # Short-window model parameters (rapid adaptation)
        self.A_short = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor parameters
        # Long window: slower forgetting
        self.lambda_long_base = 0.995
        self.lambda_long_min = 0.95
        self.lambda_long_max = 0.999
        self.current_lambda_long = self.lambda_long_base
        
        # Short window: faster forgetting for rapid adaptation
        self.lambda_short_base = 0.90
        self.lambda_short_min = 0.70
        self.lambda_short_max = 0.98
        self.current_lambda_short = self.lambda_short_base
        
        # Error tracking for adaptive lambda
        # We track the exponential moving average of squared errors per arm
        self.err_ema_long = [0.0 for _ in range(n_arms)]
        self.err_ema_short = [0.0 for _ in range(n_arms)]
        self.err_alpha = 0.1 # Learning rate for error EMA
        
        self.expected_var = 1.0 # Assume unit noise variance initially
        
        # Exploration parameter
        self.alpha_start = 2.0
        self.alpha_min = 0.5
        self.decay_rate = 0.001
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0 # 5s margin
        
        # Reset threshold multiplier
        self.reset_threshold = 2.5
        
        # Blending weight: how much to trust short vs long window
        # Higher weight means more reliance on short window for selection
        self.blend_weight = 0.7

    def _solve(self, A, b):
        """Solve linear system with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_inverse_solve(self, A, vec):
        """Solve A x = vec."""
        try:
            return np.linalg.solve(A, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, vec, rcond=None)[0]

    def _update_lambda(self, sq_error, current_lambda, lam_min, lam_max, err_ema, arm_idx):
        """Update forgetting factor for a specific arm based on error."""
        # Update EMA of squared error for this arm
        self.err_ema[arm_idx] = (1 - self.err_alpha) * self.err_ema[arm_idx] + self.err_alpha * sq_error
        avg_sq_err = self.err_ema[arm_idx]
        
        if avg_sq_err == 0:
            return current_lambda
            
        # Adaptive lambda:
        # If avg_sq_err is high, we want low lambda (more forgetting).
        # If avg_sq_err is low, we want high lambda (more memory).
        error_ratio = avg_sq_err / self.expected_var
        
        # Use a soft saturation function
        penalty = error_ratio / (1.0 + error_ratio)
        
        new_lambda = lam_max - (lam_max - lam_min) * penalty
        
        # Smooth the lambda update to avoid oscillation
        smoothed_lambda = 0.9 * current_lambda + 0.1 * new_lambda
        
        # Update expected variance estimate (slowly)
        self.expected_var = 0.99 * self.expected_var + 0.01 * (avg_sq_err + 0.1)
        
        return smoothed_lambda

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
             # Exploit best known arm using long-window model
             best_arm = 0
             best_score = -np.inf
             for a in range(self.n_arms):
                 theta = self._solve(self.A_long[a], self.b_long[a])
                 mu = theta @ context
                 if mu > best_score:
                     best_score = mu
                     best_arm = a
             return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha
        # Decay alpha over time to reduce exploration
        alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Compute predictions from both windows
            theta_short = self._solve(self.A_short[a], self.b_short[a])
            mu_short = theta_short @ context
            
            A_inv_x_short = self._get_inverse_solve(self.A_short[a], context)
            v_short = context @ A_inv_x_short
            
            theta_long = self._solve(self.A_long[a], self.b_long[a])
            mu_long = theta_long @ context
            
            A_inv_x_long = self._get_inverse_solve(self.A_long[a], context)
            v_long = context @ A_inv_x_long
            
            # Blend predictions: prioritize short window for adaptability
            mu_blend = self.blend_weight * mu_short + (1 - self.blend_weight) * mu_long
            
            # Blend uncertainty: use short window uncertainty primarily
            v_blend = self.blend_weight * v_short + (1 - self.blend_weight) * v_long
            
            # UCB Score using blended estimate
            score = mu_blend + alpha * np.sqrt(max(v_blend, 0.0))
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction and residual using short-window model
        A_short = self.A_short[arm]
        b_short = self.b_short[arm]
        theta_short = self._solve(A_short, b_short)
        mu_pred_short = theta_short @ context
        
        # Uncertainty for reset check (using short window)
        A_inv_x_short = self._get_inverse_solve(A_short, context)
        v_short = context @ A_inv_x_short
        std_pred_short = np.sqrt(max(v_short, 0.0)) + np.sqrt(self.expected_var)
        
        residual_short = reward - mu_pred_short
        sq_error_short = residual_short ** 2
        
        # Also compute residual for long window
        A_long = self.A_long[arm]
        b_long = self.b_long[arm]
        theta_long = self._solve(A_long, b_long)
        mu_pred_long = theta_long @ context
        
        residual_long = reward - mu_pred_long
        sq_error_long = residual_long ** 2
        
        # 2. Update both lambdas (per-arm adaptive)
        self.current_lambda_short = self._update_lambda(
            sq_error_short, self.current_lambda_short, 
            self.lambda_short_min, self.lambda_short_max, 
            self.err_ema_short, arm
        )
        self.current_lambda_long = self._update_lambda(
            sq_error_long, self.current_lambda_long, 
            self.lambda_long_min, self.lambda_long_max, 
            self.err_ema_long, arm
        )
        
        lam_short = self.current_lambda_short
        lam_long = self.current_lambda_long
        
        # 3. Check for Shock Reset on short window
        # If residual is significantly larger than expected standard deviation
        if abs(residual_short) > self.reset_threshold * std_pred_short:
            # Reset short window model for this arm to prioritize the new data point
            self.A_short[arm] = np.eye(self.dim) * 1.0
            self.b_short[arm] = reward * context
        else:
            # Standard RLS update with forgetting for short window
            self.A_short[arm] = lam_short * self.A_short[arm] + np.outer(context, context)
            self.b_short[arm] = lam_short * self.b_short[arm] + reward * context
        
        # 4. Update long window (no reset, just gradual adaptation)
        # Use a more conservative update to maintain stability
        self.A_long[arm] = lam_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = lam_long * self.b_long[arm] + reward * context

# EVOLVE-BLOCK-END
