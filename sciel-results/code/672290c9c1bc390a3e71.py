import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Strategy:
    Uses per-arm Bayesian Linear Regression models with an adaptive forgetting factor.
    Instead of a fixed decay (e.g., 0.995), the decay rate is modulated by the 
    recent prediction error (surprise) of the selected arm. High error triggers 
    faster forgetting (lower alpha) to adapt to changes; low error triggers slower 
    forgetting (higher alpha) to retain knowledge.
    
    Improvements:
    1. Uses np.linalg.solve instead of explicit inverse for numerical stability.
    2. Adds a Global Non-stationarity Detector. If the global prediction error is high,
       it forces a faster forgetting rate across all arms, preventing stale models in
       unpulled arms from persisting.
    3. Combines local and global error signals using a smooth 'min' approximation 
       (or direct min for alpha) to determine the effective forgetting factor.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize A with slightly larger regularization to reduce early exploration noise
        # Current program uses np.eye(dim). Let's stick to that or maybe 1.0 * I.
        # The inspiration 1 used np.eye(dim). The current program used np.eye(dim).
        # Let's keep np.eye(dim) but ensure we handle the solve correctly.
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_alpha = 0.990  # Base forgetting factor (slower decay)
        self.max_alpha = 0.999   # Max forgetting factor (very slow decay when stable)
        self.min_alpha = 0.900   # Min forgetting factor (fast decay when unstable)
        
        # State for adaptive control
        self.error_history = [0.0 for _ in range(n_arms)]
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.error_ema_alpha = 0.1  # Smoothing for error EMA
        
        # Global state
        self.global_error_ema = 0.0
        self.global_error_ema_alpha = 0.2
        
        # Exploration parameter
        # Inspiration 1 increased this to 1.2. Current was 1.0.
        # Let's use 1.1 as a middle ground, or stick to 1.0 if we want to be conservative.
        # However, the global decay mechanism might reduce the need for high UCB.
        # Let's try 1.0 first, as the global decay handles the "stale arm" issue better than pure exploration.
        self.alpha_ucb = 1.0
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0 # Safety margin for 120s limit

    def _solve_and_variance(self, A, b, context):
        """
        Compute theta = A^-1 b and variance = context^T A^-1 context.
        Uses np.linalg.solve for theta.
        For variance, we need context^T (A^-1) context.
        We can compute v = A^-1 context by solving A v = context.
        Then variance = context @ v.
        This avoids computing the full inverse matrix, which is more stable and faster.
        """
        try:
            # Solve A theta = b
            theta = np.linalg.solve(A, b)
            # Solve A v = context
            v = np.linalg.solve(A, context)
            
            mean = theta @ context
            variance = context @ v
            
            if variance < 0:
                variance = 0.0
            
            return mean, variance, theta
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse or identity if singular
            # This is unlikely with regularized A, but good for safety
            theta = np.zeros(self.d)
            mean = 0.0
            variance = 1.0
            return mean, variance, theta

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        scores = []
        context = np.asarray(context)
        
        # Dynamic exploration based on global error
        # If global error is high, we are in a non-stationary phase, explore more
        dynamic_alpha = self.alpha_ucb + 0.5 * min(self.global_error_ema, 1.0)
        
        for a in range(self.n):
            try:
                mean, variance, _ = self._solve_and_variance(self.A[a], self.b[a], context)
                
                # UCB score
                score = mean + dynamic_alpha * np.sqrt(variance)
                scores.append(score)
            except Exception:
                # Fallback if numerical issues
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            # We need the predicted reward for the selected arm
            # We can reuse the solve logic, but we already did it in select? 
            # No, select is called, then update. We need to recompute or store.
            # Storing theta for all arms in select is expensive if we don't use it.
            # Let's just recompute for the specific arm here.
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_abs = abs(error)
        except Exception:
            error_abs = 1.0

        # 2. Update Local Error EMA
        self.error_ema[arm] = (1 - self.error_ema_alpha) * self.error_ema[arm] + self.error_ema_alpha * error_abs
        
        # 3. Update Global Error EMA
        self.global_error_ema = (1 - self.global_error_ema_alpha) * self.global_error_ema + self.global_error_ema_alpha * error_abs

        # 4. Determine adaptive forgetting factor
        # We combine local and global signals.
        # Local signal: If this arm is wrong, forget it fast.
        # Global signal: If the environment is changing (global error high), forget ALL arms faster.
        
        # Local alpha calculation
        err_norm_local = min(max(self.error_ema[arm], 0.0), 1.0)
        local_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_local
        
        # Global alpha calculation
        # If global error is high, we want a lower alpha (faster forgetting) for everyone.
        # Map global error to a multiplier or a target alpha.
        # Let's say if global_error_ema is 0, global_alpha = max_alpha.
        # If global_error_ema is 1.0, global_alpha = min_alpha.
        err_norm_global = min(max(self.global_error_ema, 0.0), 1.0)
        global_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm_global
        
        # Combine: The effective alpha for the current arm is the minimum of the two?
        # No, we want the *fastest* forgetting if either signal is strong.
        # So we take the minimum alpha (which corresponds to fastest forgetting).
        adaptive_alpha = min(local_alpha, global_alpha)
        
        # Ensure alpha is within bounds
        adaptive_alpha = np.clip(adaptive_alpha, self.min_alpha, self.max_alpha)

        # 5. Update Model for the selected arm
        # A = alpha * A + xx^T
        # b = alpha * b + x * r
        self.A[arm] = adaptive_alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = adaptive_alpha * self.b[arm] + reward * context
        
        # 6. Global Update for Other Arms?
        # If global error is high, it suggests the environment has changed.
        # We should also apply a slight decay to the other arms to prevent them
        # from holding onto stale weights that are no longer relevant.
        # This is a heuristic: apply a "global decay" to unpulled arms.
        # To avoid excessive computation, we only do this if global error is significant.
        if self.global_error_ema > 0.5:
            # Apply a decay factor to all other arms
            # Use a slightly more aggressive decay for unobserved arms if global drift is detected
            # This helps "reset" the exploration of other arms.
            for a in range(self.n):
                if a != arm:
                    # Decay their A and b slightly
                    # Use a decay factor close to 1.0 but less, e.g., 0.99
                    # This is conservative to avoid destroying good estimates.
                    # Let's use a factor derived from global error.
                    # If global error is 1.0, decay by 0.95. If 0.5, decay by 0.99.
                    global_decay = 1.0 - 0.05 * self.global_error_ema
                    self.A[a] = global_decay * self.A[a]
                    self.b[a] = global_decay * self.b[a]

        # 7. Time guard
        if time.time() > self.deadline:
            # We can't really do much here other than ensure future calls are fast
            pass
