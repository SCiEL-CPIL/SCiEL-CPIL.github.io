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
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
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
        
        # Exploration parameter
        self.alpha_ucb = 1.0
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0 # Safety margin for 120s limit

    def _get_inv(self, A):
        """
        Compute inverse of A. 
        For small dimensions (dim=5), explicit inverse is fast and allows 
        direct use in UCB term calculation. 
        We use solve for theta but inv for the UCB variance term to avoid 
        repeated solves in the scoring loop if we were batching, but here 
        per-arm inv is cheap. 
        Actually, for dim=5, np.linalg.inv is O(5^3)=125 ops, very fast.
        """
        return np.linalg.inv(A)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        scores = []
        context = np.asarray(context)
        
        for a in range(self.n):
            try:
                inv_A = self._get_inv(self.A[a])
                theta = inv_A @ self.b[a]
                
                # Mean prediction
                mean = theta @ context
                
                # Variance term
                variance = context @ inv_A @ context
                if variance < 0:
                    variance = 0.0
                
                # UCB score
                score = mean + self.alpha_ucb * np.sqrt(variance)
                scores.append(score)
            except Exception:
                # Fallback if numerical issues
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            inv_A = self._get_inv(self.A[arm])
            theta = inv_A @ self.b[arm]
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_abs = abs(error)
        except Exception:
            error_abs = 1.0

        # 2. Update Error EMA
        self.error_ema[arm] = (1 - self.error_ema_alpha) * self.error_ema[arm] + self.error_ema_alpha * error_abs

        # 3. Determine adaptive forgetting factor
        # Normalize error to a scale. Assuming rewards are O(1), 
        # an error > 1.0 is significant.
        # Map error_ema to alpha.
        # Low error -> high alpha (slow forget)
        # High error -> low alpha (fast forget)
        
        # Use a simple saturation function
        # If error_ema is 0, alpha = max_alpha
        # If error_ema is >= 1.0, alpha = min_alpha
        # Linear interpolation in between, clamped.
        
        err_norm = min(max(self.error_ema[arm], 0.0), 1.0)
        # Interpolate between max_alpha and min_alpha
        # alpha = max_alpha - (max_alpha - min_alpha) * err_norm
        adaptive_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * err_norm
        
        # Ensure alpha is within bounds
        adaptive_alpha = np.clip(adaptive_alpha, self.min_alpha, self.max_alpha)

        # 4. Update Model
        # A = alpha * A + xx^T
        # b = alpha * b + x * r
        self.A[arm] = adaptive_alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = adaptive_alpha * self.b[arm] + reward * context

        # Optional: Reset other arms slightly? 
        # Usually, only the selected arm is updated in standard LinUCB.
        # However, in non-stationary settings, sometimes "global" drift is assumed.
        # The prompt says "changing arm weights". This implies each arm's weight vector 
        # might change independently or globally. 
        # Standard AEL adaptation usually updates only the pulled arm.
        # We stick to updating only the pulled arm to avoid biasing unobserved arms.
        
        # 5. Time guard
        if time.time() > self.deadline:
            # We can't really do much here other than ensure future calls are fast
            pass
