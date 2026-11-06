import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB with non-stationary handling and global model blending.
    Uses a shared global linear model blended with per-arm models for better
    sample efficiency, plus dynamic forgetting factor based on prediction errors.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Per-arm models
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.arm_counts = [0] * n_arms
        
        # Global shared model (all observations inform this)
        self.A_global = np.eye(dim) * 1.0
        self.b_global = np.zeros(dim)
        self.global_count = 0
        
        # Blend weight: how much to trust global vs per-arm
        # Starts low (trust per-arm more early on), increases as global gets more data
        self.global_blend = 0.3
        
        # Adaptive Forgetting Parameters - faster forgetting for non-stationarity
        self.lambda_min = 0.78
        self.lambda_max = 0.995
        self.error_windows = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration
        self.alpha = 0.85
        
        # Track step count
        self.t = 0

    def _get_lambda(self, arm):
        """Determine forgetting factor based on recent prediction errors."""
        if len(self.error_windows[arm]) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows[arm])
        mse = np.mean(errors**2)
        
        # Sharper transition: smaller scale factor means faster drop
        norm_mse = mse / (mse + 0.08)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            # Return best guess from global model
            theta_g = np.linalg.solve(self.A_global, self.b_global)
            return int(np.argmax([theta_g @ context] * self.n))
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        # Pre-compute global theta and inverse for blending
        try:
            inv_A_g = np.linalg.inv(self.A_global)
            theta_g = inv_A_g @ self.b_global
        except np.linalg.LinAlgError:
            inv_A_g = np.eye(self.d)
            theta_g = np.zeros(self.d)
        
        # Blend weight increases with global data
        self.global_blend = min(0.6, 0.2 + 0.01 * self.global_count)
        
        for a in range(self.n):
            try:
                inv_A = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                inv_A = np.eye(self.d)
            
            theta_arm = inv_A @ self.b[a]
            
            # Blend per-arm and global predictions
            blend = self.global_blend
            theta = (1 - blend) * theta_arm + blend * theta_g
            
            # Mean reward
            mu = theta @ context
            
            # Variance from per-arm model (conservative)
            var = context @ inv_A @ context
            
            # UCB score
            scores.append(mu + self.alpha * np.sqrt(var))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Calculate residual using the model *before* this update
        try:
            inv_A_old = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            inv_A_old = np.eye(self.d)
            
        theta_old = inv_A_old @ self.b[arm]
        pred = theta_old @ context
        residual = reward - pred
        
        # Store residual
        self.error_windows[arm].append(residual)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda(arm)
        
        # 3. Update per-arm matrices
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        self.arm_counts[arm] += 1
        
        # 4. Update global model (all observations)
        # Use a slightly slower forgetting for global (more stable)
        lam_global = max(0.95, lam)  # Global forgets slower
        self.A_global = lam_global * self.A_global + np.outer(context, context)
        self.b_global = lam_global * self.b_global + reward * context
        self.global_count += 1
