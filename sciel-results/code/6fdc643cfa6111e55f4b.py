import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin
        
        # Forgetting factors
        # Global model tracks the common trend, so it should be more stable (higher gamma)
        self.gamma_global = 0.995
        # Arm-specific offsets track rapid changes, so they should be more reactive (lower gamma)
        self.gamma_arm = 0.90
        
        # Global Linear Model
        self.A_global = np.eye(dim)
        self.b_global = np.zeros(dim)
        self.inv_A_global = np.eye(dim)
        
        # Per-Arm Linear Models (Offsets)
        # Start with proper regularization for stable initial estimates
        self.A_arm = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b_arm = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A_arm = [np.eye(dim) for _ in range(n_arms)]
        
        # Adaptive Beta for UCB
        # Use a base beta that will be scaled by a factor based on recent performance
        self.base_beta = 1.2
        self.beta_scale = 1.0
        self.beta_scale_min = 0.5
        self.beta_scale_max = 2.0
        self.beta_update_interval = 20
        
        # Tracking for best arm fallback
        self.best_arm = 0
        self.recent_rewards = np.zeros(n_arms)
        self.recent_counts = np.zeros(n_arms)
        self.ewma_alpha = 0.1
        self.ewma_values = np.zeros(n_arms)

    def _within_deadline(self):
        return time.time() < self.deadline

    def _sm_update_inverse(self, A_inv, x, gamma):
        """
        Update inverse of A given A_new = gamma * A_old + x x^T
        Returns new A_inv
        """
        # A_new = gamma * (A_old + (1/gamma) * x x^T)
        # Let M = A_old + (1/gamma) * x x^T
        # M_inv = A_inv - (A_inv * x * x^T * A_inv) / (gamma + x^T * A_inv * x)
        # A_new_inv = (1/gamma) * M_inv
        
        v = A_inv @ x
        denom = gamma + x @ v
        # Avoid division by zero
        if abs(denom) < 1e-10:
            denom = 1e-10
            
        # Sherman-Morrison term
        sm_term = np.outer(v, v) / denom
        M_inv = A_inv - sm_term
        new_inv = M_inv / gamma
        
        # Symmetrize to reduce numerical drift
        new_inv = (new_inv + new_inv.T) / 2.0
        return new_inv

    def select(self, context):
        self.t += 1
        
        if not self._within_deadline():
            return self.best_arm
            
        context = np.asarray(context)
        scores = np.zeros(self.n)
        
        # Compute global theta
        theta_global = self.inv_A_global @ self.b_global
        
        for a in range(self.n):
            # Compute arm offset theta
            theta_arm = self.inv_A_arm[a] @ self.b_arm[a]
            
            # Combined theta
            theta = theta_global + theta_arm
            
            # Mean prediction
            mean = theta @ context
            
            # Variance prediction
            # Var(x^T theta) = x^T Cov(theta) x
            # Cov(theta_global) = inv_A_global
            # Cov(theta_arm) = inv_A_arm[a]
            # Assuming independence: Cov(total) = inv_A_global + inv_A_arm[a]
            # Note: This is an approximation. The true covariance of the sum of two correlated estimators is different.
            # However, in a hierarchical model, this is a standard way to combine uncertainties.
            
            # Calculate variance scalar
            # x^T (inv_A_global + inv_A_arm[a]) x
            v_global = context @ self.inv_A_global
            v_arm = context @ self.inv_A_arm[a]
            var = context @ (v_global + v_arm)
            
            # Ensure variance is positive
            if var < 1e-10:
                var = 1e-10
                
            # UCB Score
            # Use adaptive beta for better exploration-exploitation balance
            scores[a] = mean + self.base_beta * self.beta_scale * np.sqrt(var)
            
        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        self.t += 1
        
        if not self._within_deadline():
            return
            
        context = np.asarray(context)
        
        # 1. Update Global Model
        # A_global_new = gamma_global * A_global_old + x x^T
        # We update the inverse directly using SM formula
        self.inv_A_global = self._sm_update_inverse(self.inv_A_global, context, self.gamma_global)
        self.b_global = self.gamma_global * self.b_global + reward * context
        
        # 2. Update Per-Arm Model
        self.inv_A_arm[arm] = self._sm_update_inverse(self.inv_A_arm[arm], context, self.gamma_arm)
        self.b_arm[arm] = self.gamma_arm * self.b_arm[arm] + reward * context
        
        # 3. Update Tracking Statistics
        self.ewma_values[arm] = (1 - self.ewma_alpha) * self.ewma_values[arm] + self.ewma_alpha * reward
        self.recent_counts[arm] += 1
        
        # Update best arm based on EWMA for deadline fallback
        if self.t % 10 == 0:
             valid_arms = [a for a in range(self.n) if self.recent_counts[a] > 0]
             if valid_arms:
                 self.best_arm = max(valid_arms, key=lambda a: self.ewma_values[a])
        
        # 4. Adapt beta scale based on recent reward variance
        if self.t % self.beta_update_interval == 0 and self.t > 100:
            # If rewards are stable (low variance), reduce exploration
            # If rewards are volatile (high variance), increase exploration
            recent_var = np.var(self.ewma_values)
            # Normalize variance to [0, 1] range approximately
            var_norm = min(1.0, max(0.0, recent_var / 1.0))
            # Higher variance -> higher beta_scale (more exploration)
            self.beta_scale = self.beta_scale_min + (self.beta_scale_max - self.beta_scale_min) * var_norm
