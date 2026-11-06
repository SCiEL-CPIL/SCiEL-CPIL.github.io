import numpy as np
import time
from collections import deque

class Policy:
    """
    Global linear model with arm-specific bias terms for non-stationary linear contextual bandits.
    
    Strategy:
    - Single global linear model shared across all arms to leverage cross-arm correlations.
    - Arm-specific bias terms to capture arm-specific offsets.
    - Exponential forgetting factor adapted to prediction error.
    - Dynamic exploration parameter (alpha) based on regret/error signals.
    - Cached matrix inverses for efficient select().
    - Deadline guard to ensure termination within time limits.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        ridge = 1.0
        
        # Global model
        self.A = np.eye(dim) * ridge
        self.b = np.zeros(dim)
        self.inv_A = np.eye(dim) / ridge
        self.theta = np.zeros(dim)
        
        # Arm-specific bias terms
        self.arm_bias = np.zeros(n_arms)
        self.arm_bias_var = np.ones(n_arms) * ridge
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.88
        self.lambda_max = 0.99
        self.error_windows = deque(maxlen=50)
        
        # Exploration
        self.alpha_base = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.0
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 120s limit
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=100)

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_lambda(self):
        """Determine forgetting factor based on recent errors."""
        if len(self.error_windows) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows)
        mse = np.mean(errors**2)
        
        # If MSE is small, we are stable, so forget slowly (lambda close to 1)
        # If MSE is large, we are unstable, so forget fast (lambda lower)
        
        # Scale factor: assume typical reward variance is around 1.0
        # Using a saturating function to map MSE to lambda
        scale = 0.1
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance with trend awareness."""
        if len(self.global_errors) < 10:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        # Also track recent trend: compare last 5 errors to previous 5
        if len(errors) >= 10:
            recent = errors[-5:]
            prev = errors[-10:-5]
            trend = np.mean(recent**2) - np.mean(prev**2)
        else:
            trend = 0.0
        
        # Base target from MSE
        base_target = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.15))
        
        # Adjust for trend: if errors are increasing, boost exploration
        trend_boost = 0.1 * np.clip(trend, -0.5, 0.5)
        target_alpha = base_target + trend_boost
        
        # Smoothly move towards target with faster response
        self.alpha = 0.85 * self.alpha + 0.15 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_global_cache(self, context, reward):
        """Recompute inverse and theta for the global model using SMW."""
        lam = self._get_lambda()
        inv_old = self.inv_A
        x = context
        r = reward
        
        # Compute x^T inv_old x
        tmp = inv_old @ x
        denom = lam + x @ tmp
        
        if abs(denom) < 1e-10:
            # Fallback to full inverse
            try:
                self.inv_A = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self.inv_A = np.eye(self.d)
        else:
            # SMW update
            self.inv_A = (1.0 / lam) * (inv_old - np.outer(tmp, tmp) / denom)
        
        self.theta = self.inv_A @ self.b

    def select(self, context):
        if not self._check_deadline():
            # If we are near the deadline, just pick the arm with highest mean prediction
            # to avoid expensive computations or just pick a random one?
            # Better to pick the best known arm.
            scores = []
            for a in range(self.n):
                mu = self.theta @ context + self.arm_bias[a]
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta @ context + self.arm_bias[a]
            # var = context @ inv_A @ context + arm_bias_var[a]
            var = context @ self.inv_A @ context + self.arm_bias_var[a]
            # Clip variance to avoid negative due to numerical issues
            var = max(var, 0.0)
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Calculate residual using the model *before* this update
        # We use the cached theta which is current
        pred = self.theta @ context + self.arm_bias[arm]
        residual = reward - pred
        
        # Store residual
        self.error_windows.append(residual)
        self.global_errors.append(residual)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda()
        
        # 3. Update global matrices
        self.A = lam * self.A + np.outer(context, context)
        self.b = lam * self.b + reward * context
        
        # 4. Update arm-specific bias
        self.arm_bias[arm] = lam * self.arm_bias[arm] + reward
        self.arm_bias_var[arm] = lam * self.arm_bias_var[arm] + 1.0
        
        # 5. Update cache for global model using SMW
        self._update_global_cache(context, reward)
        
        # 6. Update alpha
        self._update_alpha()
