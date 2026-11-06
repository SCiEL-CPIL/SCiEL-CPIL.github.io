"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive LinUCB for non-stationary linear bandits.
    Uses per-arm models with an adaptive forgetting factor and adaptive exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        self.base_lambda = 0.95
        self.min_lambda = 0.50  # More aggressive forgetting
        self.max_lambda = 0.99
        
        # Buffer for recent errors per arm
        self.error_buffers = [[] for _ in range(n_arms)]
        self.error_window_size = 5  # Shorter window for faster reaction
        
        # UCB exploration parameter
        self.alpha = 1.0
        self.base_alpha = 1.0
        self.max_alpha = 5.0
        
        # Store last predictions for error calculation
        self._last_preds = [0.0] * n_arms
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 145.0

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context)
        scores = []
        preds = []
        
        for a in range(self.n):
            try:
                # Use solve for theta: A * theta = b
                theta = np.linalg.solve(self.A[a], self.b[a])
                mu = theta @ context
                preds.append(mu)
                
                # Compute variance: x^T A^{-1} x
                # Let v = A^{-1} x, then var = x^T v
                v = np.linalg.solve(self.A[a], context)
                var = context @ v
                std = np.sqrt(max(0, var))
                
                # UCB score
                score = mu + self.alpha * std
                scores.append(score)
            except np.linalg.LinAlgError:
                # Fallback
                scores.append(0.0)
                preds.append(0.0)
                
        self._last_preds = preds
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() > self._deadline:
            return

        context = np.asarray(context)
        
        # 1. Calculate prediction error
        pred = self._last_preds[arm]
        error = abs(reward - pred)
        
        # 2. Update error buffer
        self.error_buffers[arm].append(error)
        if len(self.error_buffers[arm]) > self.error_window_size * 2:
            self.error_buffers[arm] = self.error_buffers[arm][-self.error_window_size:]
            
        recent_errors = self.error_buffers[arm][-self.error_window_size:]
        mean_err = np.mean(recent_errors) if recent_errors else 0.0
        
        # 3. Adapt Exploration Parameter (Alpha)
        # Increase alpha when errors are high to explore more
        # Normalize error: assume typical error scale is ~0.1-0.5
        # If mean_err is 0.5, factor = 1 + 0.5/0.1 * 0.5 = 3.5 (capped)
        # Let's use a softer mapping
        err_scale = 0.2
        normalized_err = mean_err / err_scale
        alpha_boost = min(2.0, normalized_err) # Cap boost at 2x
        self.alpha = self.base_alpha * (1.0 + alpha_boost)
        
        # 4. Determine Forgetting Factor (Lambda)
        if not recent_errors:
            lam = self.base_lambda
        else:
            # Map error to lambda
            # High error -> low lambda
            # Low error -> high lambda
            
            # Thresholds
            low_err_thresh = 0.05
            high_err_thresh = 0.5
            
            if mean_err <= low_err_thresh:
                lam = self.max_lambda
            elif mean_err >= high_err_thresh:
                lam = self.min_lambda
            else:
                # Linear interpolation
                t = (mean_err - low_err_thresh) / (high_err_thresh - low_err_thresh)
                lam = self.max_lambda - t * (self.max_lambda - self.min_lambda)
                
        # 5. Update Model
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
