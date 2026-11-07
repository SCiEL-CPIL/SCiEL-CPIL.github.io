"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Rank-1 Covariance Adaptation (CMA-ES lite)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        # Start near center to avoid boundary issues initially, or random?
        # Bent cigar is usually centered at 0 but shifted/rotated.
        # Random start in [-5, 5] is safer for unknown shifts.
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()
        
        sigma = 1.0
        C = np.eye(self.dim)
        
        # 1/5 rule parameters
        window_size = 10
        window = []
        
        calls = 1
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C = np.eye(self.dim)
                L = np.linalg.cholesky(C)
                
            z = rng.standard_normal(self.dim)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                
                # Update Covariance (Rank-1)
                # The direction of the step in the standardized space is z.
                # The actual step is sigma * L * z.
                # We want C to represent the covariance of the direction L*z / sigma?
                # Standard CMA-ES updates C based on the standardized step.
                # Here, we update C towards the outer product of the standardized step direction.
                # However, since we are using (1+1) and not population, the update is simpler.
                # We want the distribution to align with the successful direction.
                # The successful direction in the "C-space" is z.
                # So we update C towards z z^T.
                
                mu = 0.3  # Learning rate for C
                C = (1 - mu) * C + mu * np.outer(z, z)
                
                # Normalize C to maintain volume (trace = dim)
                # This separates shape adaptation from scale adaptation (sigma)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)
                    
                window.append(1)
            else:
                window.append(0)
            
            # Adapt Sigma using 1/5 rule
            if len(window) >= window_size:
                window.pop(0)
                success_rate = sum(window) / window_size
                # If success rate is too high, increase sigma; too low, decrease
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                else:
                    sigma *= 1.0
                
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-4, 10.0)
            
            # Optional: Clip C eigenvalues to prevent numerical instability
            # In 5D, this is rarely needed with trace normalization, but good for safety
            if calls % 50 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.1, 10.0)
                C = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
                # Renormalize trace
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)

        return best_x, best_fx
# EVOLVE-BLOCK-END
