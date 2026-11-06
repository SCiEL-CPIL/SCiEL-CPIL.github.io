"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive LinUCB for non-stationary linear bandits.
    Uses per-arm models with an adaptive forgetting factor that reacts
    to prediction errors to track changing arm weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        # A is the covariance matrix (inverse), b is the moment vector
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        # We want to forget fast when errors are high, slow when errors are low.
        self.base_lambda = 0.95
        self.min_lambda = 0.85
        self.max_lambda = 0.99
        
        # Buffer for recent errors per arm to determine local stability
        self.error_buffers = [[] for _ in range(n_arms)]
        self.error_window_size = 20
        
        # UCB exploration parameter
        self.alpha = 1.0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 145.0  # 5s safety margin before 150s kill
        
        # Store last prediction for each arm to compute error in update
        self._last_preds = [0.0] * self.n

    def _get_lambda(self, arm):
        """
        Calculate the adaptive forgetting factor for a specific arm
        based on recent prediction errors.
        """
        buf = self.error_buffers[arm]
        if not buf:
            return self.base_lambda
            
        # Use the mean absolute error in the recent window
        # If the window is full, take the last 'window_size' errors
        recent = buf[-self.error_window_size:]
        mean_err = np.mean(recent)
        
        # Map error magnitude to lambda.
        # High error -> low lambda (forget fast)
        # Low error -> high lambda (remember)
        
        # Heuristic scaling: 
        # If mean_err is very small, lambda approaches max.
        # If mean_err is large, lambda approaches min.
        
        # Normalize error. Assuming rewards are in a reasonable range (e.g., 0-1 or -1-1),
        # an error of 0.1 is significant, 0.01 is small.
        # We use a soft cap.
        err_scale = 0.1 
        normalized_err = mean_err / err_scale
        
        # Sigmoid-like mapping or linear clamp
        # Let's use a simple linear decay from max to min over a range of errors.
        # If normalized_err > 1, we are in "high error" zone.
        # If normalized_err < 0.1, we are in "low error" zone.
        
        if normalized_err <= 0.1:
            return self.max_lambda
        elif normalized_err >= 1.0:
            return self.min_lambda
        else:
            # Linear interpolation
            t = (normalized_err - 0.1) / (1.0 - 0.1)
            return self.max_lambda - t * (self.max_lambda - self.min_lambda)

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            # If we are past deadline, we should just return a random arm
            # to avoid hanging, though the evaluator kills the process anyway.
            # This is a safety net.
            return int(np.random.randint(self.n))

        context = np.asarray(context)
        scores = []
        preds = [0.0] * self.n
        
        for a in range(self.n):
            # Compute theta and UCB
            try:
                # Use Cholesky decomposition for better numerical stability and speed
                # A is symmetric positive definite (initialized as I, updated with outer products)
                try:
                    L = np.linalg.cholesky(self.A[a])
                    # Solve L^T L theta = b  =>  L theta = b (after first solve)
                    # Actually, A = L @ L.T
                    # We need inv(A) @ b.
                    # Solve L.T y = b, then L theta = y
                    y = np.linalg.solve(L.T, self.b[a])
                    theta = np.linalg.solve(L, y)
                    
                    # Mean prediction
                    mu = theta @ context
                    preds[a] = mu
                    
                    # Variance term
                    # var = context.T @ inv(A) @ context
                    # inv(A) = inv(L.T) @ inv(L)
                    # Let z = inv(L) @ context
                    # Then var = z @ z
                    z = np.linalg.solve(L, context)
                    var = np.dot(z, z)
                    std = np.sqrt(max(0, var))
                    
                    # UCB score
                    score = mu + self.alpha * std
                    scores.append(score)
                except np.linalg.LinAlgError:
                    # Fallback to inv if Cholesky fails
                    inv_A = np.linalg.inv(self.A[a])
                    theta = inv_A @ self.b[a]
                    mu = theta @ context
                    preds[a] = mu
                    var = context @ inv_A @ context
                    std = np.sqrt(max(0, var))
                    score = mu + self.alpha * std
                    scores.append(score)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                scores.append(0.0)
                preds[a] = 0.0
                
        arm = int(np.argmax(scores))
        
        # Store predictions for error calculation in update
        self._last_preds = preds
        
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Deadline check
        if time.time() > self._deadline:
            return

        # 1. Update model with adaptive forgetting
        lam = self._get_lambda(arm)
        
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # 2. Update error buffer for adaptive lambda calculation
        # Use the prediction stored in select
        pred = self._last_preds[arm]
        error = abs(reward - pred)
        
        # Append error to buffer
        self.error_buffers[arm].append(error)
        
        # Keep buffer size limited
        if len(self.error_buffers[arm]) > self.error_window_size * 2:
            self.error_buffers[arm] = self.error_buffers[arm][-self.error_window_size:]

# EVOLVE-BLOCK-END
