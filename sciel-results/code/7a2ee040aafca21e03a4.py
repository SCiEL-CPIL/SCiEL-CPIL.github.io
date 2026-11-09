"""
Non-stationary Linear Contextual Bandit Policy.

Uses a multi-timescale Linear UCB approach:
1. Short-window model (fast adaptation) for primary arm selection.
2. Long-window model (stable baseline) for regularization/backup.
3. Adaptive forgetting factors for both windows.
4. Maintains inverse matrices for efficient uncertainty computation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Short-window models (fast) - track A, b, and A_inv
        self.A_short = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv_short = [np.eye(dim) for _ in range(n_arms)]
        
        # Long-window models (slow)
        self.A_long = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv_long = [np.eye(dim) for _ in range(n_arms)]
        
        # Forgetting factors
        self.lambda_short_base = 0.95
        self.lambda_short_min = 0.70
        self.lambda_short_max = 0.99
        self.lambda_short = self.lambda_short_base
        
        self.lambda_long_base = 0.995
        self.lambda_long_min = 0.95
        self.lambda_long_max = 0.9999
        self.lambda_long = self.lambda_long_base
        
        # Error tracking
        self.error_buffer_short = np.zeros(30)
        self.error_idx_short = 0
        self.error_count_short = 0
        self.expected_var_short = 1.0
        
        self.error_buffer_long = np.zeros(50)
        self.error_idx_long = 0
        self.error_count_long = 0
        self.expected_var_long = 1.0
        
        # Exploration
        self.alpha_start = 2.0
        self.alpha_min = 0.3
        self.decay_rate = 0.003
        
        # Time
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0
        
        # Reset thresholds
        self.reset_threshold = 2.5

    def _update_lambda_short(self, sq_error):
        """Update short-window forgetting factor."""
        self.error_buffer_short[self.error_idx_short] = sq_error
        self.error_idx_short = (self.error_idx_short + 1) % len(self.error_buffer_short)
        if self.error_count_short < len(self.error_buffer_short):
            self.error_count_short += 1
            
        if self.error_count_short == 0:
            return
            
        avg_sq_err = np.mean(self.error_buffer_short[:self.error_count_short])
        error_ratio = avg_sq_err / (self.expected_var_short + 1e-8)
        penalty = error_ratio / (1.0 + error_ratio)
        new_lambda = self.lambda_short_max - (self.lambda_short_max - self.lambda_short_min) * penalty
        self.lambda_short = 0.9 * self.lambda_short + 0.1 * new_lambda
        self.expected_var_short = 0.99 * self.expected_var_short + 0.01 * (avg_sq_err + 0.1)

    def _update_lambda_long(self, sq_error):
        """Update long-window forgetting factor."""
        self.error_buffer_long[self.error_idx_long] = sq_error
        self.error_idx_long = (self.error_idx_long + 1) % len(self.error_buffer_long)
        if self.error_count_long < len(self.error_buffer_long):
            self.error_count_long += 1
            
        if self.error_count_long == 0:
            return
            
        avg_sq_err = np.mean(self.error_buffer_long[:self.error_count_long])
        error_ratio = avg_sq_err / (self.expected_var_long + 1e-8)
        penalty = error_ratio / (1.0 + error_ratio)
        new_lambda = self.lambda_long_max - (self.lambda_long_max - self.lambda_long_min) * penalty
        self.lambda_long = 0.95 * self.lambda_long + 0.05 * new_lambda
        self.expected_var_long = 0.999 * self.expected_var_long + 0.001 * (avg_sq_err + 0.1)

    def _reset_arm_short(self, arm, context, reward):
        """Reset short-window model for an arm."""
        self.A_short[arm] = np.eye(self.dim) * 1.0
        self.b_short[arm] = reward * context
        self.A_inv_short[arm] = np.eye(self.dim)

    def _reset_arm_long(self, arm, context, reward):
        """Reset long-window model for an arm."""
        self.A_long[arm] = np.eye(self.dim) * 1.0
        self.b_long[arm] = reward * context
        self.A_inv_long[arm] = np.eye(self.dim)

    def select(self, context):
        # Deadline check - use greedy on short model
        if time.time() > self.deadline:
             best_arm = 0
             best_score = -np.inf
             ctx = np.asarray(context, dtype=np.float64)
             for a in range(self.n_arms):
                 theta = self.A_inv_short[a] @ self.b_short[a]
                 mu = theta @ ctx
                 if mu > best_score:
                     best_score = mu
                     best_arm = a
             return best_arm

        self.t += 1
        ctx = np.asarray(context, dtype=np.float64)
        
        alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Short-window model
            A_inv_s = self.A_inv_short[a]
            b_s = self.b_short[a]
            theta_s = A_inv_s @ b_s
            mu_s = theta_s @ ctx
            
            # Uncertainty from short model: v = ctx^T A_inv ctx
            v_s = ctx @ (A_inv_s @ ctx)
            v_s = max(v_s, 0.0)
            
            # Long-window model
            A_inv_l = self.A_inv_long[a]
            b_l = self.b_long[a]
            theta_l = A_inv_l @ b_l
            mu_l = theta_l @ ctx
            
            v_l = ctx @ (A_inv_l @ ctx)
            v_l = max(v_l, 0.0)
            
            # Inverse-variance blending: weight by 1/(1+v)
            w_s = 1.0 / (1.0 + v_s)
            w_l = 1.0 / (1.0 + v_l)
            w_sum = w_s + w_l
            w_s_norm = w_s / w_sum
            w_l_norm = w_l / w_sum
            
            mu_blend = w_s_norm * mu_s + w_l_norm * mu_l
            
            # Combined uncertainty: weighted combination
            v_blend = w_s_norm * v_s + w_l_norm * v_l
            
            score = mu_blend + alpha * np.sqrt(v_blend)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        
        # --- Update short-window model ---
        A_inv_s = self.A_inv_short[arm]
        b_s = self.b_short[arm]
        
        # Compute prediction and uncertainty before update
        theta_s = A_inv_s @ b_s
        mu_pred_s = theta_s @ ctx
        v_s = ctx @ (A_inv_s @ ctx)
        v_s = max(v_s, 0.0)
        std_pred_s = np.sqrt(v_s) + np.sqrt(self.expected_var_short)
        
        residual_s = reward - mu_pred_s
        sq_error_s = residual_s ** 2
        
        self._update_lambda_short(sq_error_s)
        lam_s = self.lambda_short
        
        # Check for shock reset on short model
        if abs(residual_s) > self.reset_threshold * std_pred_s:
            self._reset_arm_short(arm, ctx, reward)
        else:
            # Update A and b with forgetting
            outer = np.outer(ctx, ctx)
            self.A_short[arm] = lam_s * self.A_short[arm] + outer
            self.b_short[arm] = lam_s * b_s + reward * ctx
            
            # Update A_inv using Sherman-Morrison-like approach
            # New A = lam * A_old + xx^T
            # We want (lam * A_old + xx^T)^{-1}
            # = (1/lam) * (A_old + (1/lam) xx^T)^{-1}
            # Use Sherman-Morrison: (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
            # Here A = lam * A_old, so A^{-1} = (1/lam) * A_old^{-1}
            # New A = lam * A_old + xx^T = lam * (A_old + (1/lam) xx^T)
            # (New A)^{-1} = (1/lam) * (A_old + (1/lam) xx^T)^{-1}
            # Using SM on A_old: (A_old + (1/lam) xx^T)^{-1} = A_old^{-1} - A_old^{-1} (x/sqrt(lam)) (x/sqrt(lam))^T A_old^{-1} / (1 + (1/lam) x^T A_old^{-1} x)
            # = A_old^{-1} - (1/lam) A_old^{-1} xx^T A_old^{-1} / (1 + (1/lam) x^T A_old^{-1} x)
            # So (New A)^{-1} = (1/lam) * A_old^{-1} - (1/lam^2) A_old^{-1} xx^T A_old^{-1} / (1 + (1/lam) x^T A_old^{-1} x)
            
            A_inv_old = A_inv_s
            x = ctx
            lam = lam_s
            
            # Compute x^T A_inv x
            Ax = A_inv_old @ x
            xAx = x @ Ax  # scalar
            
            denom = 1.0 + xAx / lam
            if abs(denom) < 1e-12:
                # Fallback: recompute inverse
                try:
                    self.A_inv_short[arm] = np.linalg.inv(self.A_short[arm])
                except:
                    self.A_inv_short[arm] = np.eye(self.dim)
            else:
                # (1/lam) * A_inv_old - (1/lam^2) * A_inv_old @ x @ x^T @ A_inv_old / denom
                term1 = (1.0 / lam) * A_inv_old
                # A_inv_old @ x = Ax
                # x^T @ A_inv_old = Ax^T
                # So A_inv_old @ x @ x^T @ A_inv_old = Ax @ Ax^T
                term2 = (1.0 / (lam * lam)) * (np.outer(Ax, Ax) / denom)
                self.A_inv_short[arm] = term1 - term2
        
        # --- Update long-window model ---
        A_inv_l = self.A_inv_long[arm]
        b_l = self.b_long[arm]
        
        theta_l = A_inv_l @ b_l
        mu_pred_l = theta_l @ ctx
        v_l = ctx @ (A_inv_l @ ctx)
        v_l = max(v_l, 0.0)
        std_pred_l = np.sqrt(v_l) + np.sqrt(self.expected_var_long)
        
        residual_l = reward - mu_pred_l
        sq_error_l = residual_l ** 2
        
        self._update_lambda_long(sq_error_l)
        lam_l = self.lambda_long
        
        # Less aggressive reset for long model
        if abs(residual_l) > 4.0 * std_pred_l:
            self._reset_arm_long(arm, ctx, reward)
        else:
            outer = np.outer(ctx, ctx)
            self.A_long[arm] = lam_l * self.A_long[arm] + outer
            self.b_long[arm] = lam_l * b_l + reward * ctx
            
            A_inv_old = A_inv_l
            x = ctx
            lam = lam_l
            
            Ax = A_inv_old @ x
            xAx = x @ Ax
            
            denom = 1.0 + xAx / lam
            if abs(denom) < 1e-12:
                try:
                    self.A_inv_long[arm] = np.linalg.inv(self.A_long[arm])
                except:
                    self.A_inv_long[arm] = np.eye(self.dim)
            else:
                term1 = (1.0 / lam) * A_inv_old
                term2 = (1.0 / (lam * lam)) * (np.outer(Ax, Ax) / denom)
                self.A_inv_long[arm] = term1 - term2

# EVOLVE-BLOCK-END
