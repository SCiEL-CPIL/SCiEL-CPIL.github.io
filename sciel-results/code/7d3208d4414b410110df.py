"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # CMA-ES parameters
        dim = self.dim
        budget = self.budget
        
        # Initial mean vector
        # Start near center but slightly offset to avoid symmetry issues if any, 
        # though random is fine. Lunacek has a known structure.
        x0 = np.zeros(dim)
        
        # Initial step size
        # Heuristic for CMA-ES initialization
        sigma = 1.0 
        
        # Population size
        # Standard recommendation: lambda = 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + 3*1.6 = ~8.8 -> 9 or 10. 
        # Let's use lambda = 10 for robustness in low dim.
        lambda_ = 10
        
        # Selection size (top half usually)
        mu = lambda_ // 2
        
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute sqrt of C for sampling
        # We will update C each generation, so we need to recompute sqrt
        # or use Cholesky. For dim=5, Cholesky is cheap.
        
        def sqrtm(C):
            # Use eigen decomposition for stability in small dims
            # Or Cholesky if positive definite
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen if not PD (shouldn't happen often with CMA-ES)
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-10)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        
        B = sqrtm(C)
        D = np.sqrt(np.diag(C))
        
        # Initial function call
        fbest = func(x0)
        best_x = x0.copy()
        
        calls = 1
        gen = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            for i in range(lambda_):
                # Sample from N(0, I)
                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                # Clip to bounds [-5, 5]
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate population
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = fs[order]
            
            # Update best
            if fs[0] < fbest:
                fbest = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_new = sum(w[i] * ys[i] for i in range(mu))
            
            # Update step size control (sigma)
            # Calculate ps
            # We need the inverse square root of C from the previous step
            # For simplicity and speed in dim=5, we approximate or use the current B
            # Proper CMA-ES uses C^{-1/2}. 
            # Let's compute C^{-1/2} via eigen
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))
            
            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (pc @ pc + (1 - (1 - c_c)**2) * C) \
                + c_mu * rankmu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x0 = x_new
            gen += 1
            
            # Update B and D for next iteration
            B = sqrtm(C)
            
        return best_x, fbest
# EVOLVE-BLOCK-END
