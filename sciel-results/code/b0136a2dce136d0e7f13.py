"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive LinUCB with Cholesky decomposition and adaptive exploration.
    Structural changes:
    1. Cholesky decomposition for stable and fast theta estimation.
    2. Adaptive alpha: increases exploration when recent prediction errors are high.
    3. Robust deadline guard.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0  # Regularization
        
        # Initialize A and b for each arm
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor: 0.98 is a good balance for non-stationary tasks
        # It allows the model to adapt to changes within ~50 steps
        self.beta = 0.98
        
        # Adaptive alpha parameters
        self.alpha_base = 1.0
        self.alpha_max = 3.0
        self.alpha_min = 0.5
        self.alpha = self.alpha_base
        
        # Error tracking for adaptive alpha
        self.recent_errors = []
        self.error_window_size = 10
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 5s safety margin
        
        # Pre-allocate storage for Cholesky factors to reduce overhead? 
        # For small dim, recomputing is fine.

    def _solve_chol(self, L, b):
        """Solve L L^T x = b using Cholesky factor L."""
        # Forward substitution Ly = b
        y = np.linalg.solve(L, b)
        # Backward substitution L^T x = y
        x = np.linalg.solve(L.T, y)
        return x

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0
            
        context = np.asarray(context)
        scores = []
        
        for a in range(self.n):
            try:
                # Compute Cholesky factor
                L = np.linalg.cholesky(self.A[a])
                
                # Solve for theta: A theta = b  =>  L L^T theta = b
                theta = self._solve_chol(L, self.b[a])
                
                # Compute variance: context^T A^{-1} context
                # A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
                # Let v = L^{-1} context, then variance = v^T v
                # Solve L v = context
                v = np.linalg.solve(L, context)
                variance = np.dot(v, v)
                
                # Ensure variance is non-negative due to numerical precision
                variance = max(variance, 0.0)
                
                # Predicted reward
                pred = np.dot(theta, context)
                
                # UCB score
                ucb = pred + self.alpha * np.sqrt(variance)
                scores.append(ucb)
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (shouldn't happen with regularization)
                scores.append(0.0)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return
            
        context = np.asarray(context)
        
        # 1. Adaptive Alpha Update
        # We need the predicted reward before updating the model to calculate error
        # However, select() already ran. We can re-estimate or store the last prediction.
        # For efficiency, we'll estimate the current model's prediction for this context.
        try:
            L = np.linalg.cholesky(self.A[arm])
            theta = self._solve_chol(L, self.b[arm])
            pred = np.dot(theta, context)
            error = reward - pred
        except Exception:
            error = 0.0
            
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window_size:
            self.recent_errors.pop(0)
            
        # If recent errors are high, increase alpha to explore more
        if len(self.recent_errors) > 5:
            mean_abs_error = np.mean(np.abs(self.recent_errors))
            # Scale alpha based on error magnitude
            # If error is large, we are likely in a non-stationary phase or high noise
            # Increase alpha to widen confidence bounds
            self.alpha = np.clip(self.alpha_base * (1 + mean_abs_error), self.alpha_min, self.alpha_max)
            
        # 2. Model Update with Forgetting
        # A_new = beta * A_old + xx^T
        # b_new = beta * b_old + r * x
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
        
        # 3. Ensure numerical stability (symmetrize A)
        # This helps prevent drift from breaking Cholesky
        self.A[arm] = (self.A[arm] + self.A[arm].T) / 2.0
# EVOLVE-BLOCK-END
