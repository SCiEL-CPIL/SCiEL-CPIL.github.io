"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Covariance Adaptation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        C = np.eye(self.dim)
        
        # Parameters for covariance update
        # mu controls how fast we adapt to the new direction
        # A smaller mu means slower adaptation, more stable
        # For (1+1), we can use a simple exponential moving average of the outer product
        # Or just replace C with the outer product of the step, scaled appropriately.
        # A robust simple update: C = (1 - alpha) * C + alpha * (step / sigma) @ (step / sigma).T
        # We need to ensure C remains positive definite.
        
        alpha = 0.5  # Learning rate for covariance
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate step from multivariate normal with covariance C
            # We can use Cholesky decomposition, but for dim=5, it's cheap.
            # However, if C becomes ill-conditioned or singular, this can fail.
            # A simpler robust approach for small dim: use SVD or just diagonal adaptation?
            # Full CMA-ES is complex. Let's try a simplified covariance adaptation.
            
            # To avoid numerical issues with Cholesky, we can use a diagonal covariance matrix
            # or just adapt the step size per dimension? No, that's not enough for rotation.
            
            # Let's stick to a simplified CMA-like update.
            # z ~ N(0, I)
            z = rng.standard_normal(self.dim)
            
            # Project z to the shape of C.
            # If we maintain C, we need to sample from N(0, C).
            # For dim=5, Cholesky is fine if C is PD.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity or regularize
                C = np.eye(self.dim)
                L = np.eye(self.dim)
            
            step = L @ z
            y = np.clip(x + sigma * step, -5, 5)
            fy = func(y)
            
            if fy < fx:
                # Success
                x, fx = y, fy
                sigma *= 1.05  # Slight increase, conservative
                
                # Update Covariance
                # The actual step taken was sigma * step.
                # We want C to reflect the direction of successful steps.
                # Normalized step direction: step / ||step|| ? No, we want the covariance of the steps.
                # If we assume the step is the random vector, then the covariance of the step is sigma^2 * C.
                # We want to update C such that it aligns with the direction of improvement.
                # A common simple update: C = (1 - alpha) * C + alpha * (step @ step.T)
                # This keeps C scaled to the magnitude of the step.
                # To keep C normalized (trace=dim or similar), we might need scaling.
                # Let's just update C and renormalize its trace to dim to keep sigma meaningful.
                
                outer = np.outer(step, step)
                C = (1 - alpha) * C + alpha * outer
                
                # Renormalize C so that trace(C) = dim
                # This ensures that sigma remains the average step length
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)
                    
            else:
                # Failure
                sigma *= 0.95  # Slight decrease
                
        return x, fx
# EVOLVE-BLOCK-END
