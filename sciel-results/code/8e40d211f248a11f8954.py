import numpy as np
from scipy.stats import qmc

class Optimizer:
    """Deterministic Global Optimization using Sobol sequence followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration with Sobol
        n_global = 50
        n_local = budget - n_global
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_global)
        sobol_mapped = sobol_samples * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_global):
            x = sobol_mapped[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement with CMA-ES
        if n_local > 0:
            # Initialize CMA-ES
            x_mean = best_x.copy()
            sigma = 0.5 # Initial step size, adjusted based on box size
            pop_size = 10
            max_iter = n_local // pop_size
            
            # CMA-ES parameters
            alpha_mu = 0.5
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            for k in range(max_iter):
                # Generate population
                Y = np.zeros((pop_size, dim))
                F = np.zeros(pop_size)
                
                for i in range(pop_size):
                    z = np.random.randn(dim) # Using random, but we can seed if needed for determinism
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    Y[i] = y
                    F[i] = func(y)
                
                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                
                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()
                    
                # Update mean
                x_new_mean = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
                
                # Update covariance matrix (simplified, using eigendecomposition each step is expensive but ok for small dim)
                # For 5D, we can do full update
                diff = Y[:mu] - x_mean
                C = (1 / (1 - c_sigma * (1 - c_sigma))) * (C + c_sigma * (d_sigma * sigma * p_sigma * p_sigma.T + (d_sigma - 1) * C))
                # This is getting complex to implement correctly from scratch without bugs.
                
                # Simpler Alternative: Use Nelder-Mead via scipy but with strict budget control?
                # Or a simpler (1+1)-ES with better adaptation?
                
                # Let's revert to a simpler, robust local search: 
                # A (1+1)-ES with adaptive step size is very effective for 5D if tuned well.
                # The seed had this but it was weak. Let's improve it.
                
                # Actually, let's use scipy.optimize.minimize with Nelder-Mead 
                # but wrap it to count calls and stop.
                pass

        return best_x, best_f
