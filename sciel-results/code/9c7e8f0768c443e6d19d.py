"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with coarse grid initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 300s. We stop at 290s.
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Coarse Grid Search
        # A 3x3x3x3x3 grid covers the domain [-5, 5]^5 systematically.
        # Total points: 3^5 = 243.
        n_grid = 3
        grid_points = np.array(np.meshgrid(*([np.linspace(-5, 5, n_grid)] * self.dim))).T.reshape(-1, self.dim)
        
        best_x = None
        best_f = np.inf
        samples = []
        
        for p in grid_points:
            if time.time() > deadline:
                break
            fp = func(p)
            samples.append((fp, p))
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
                
        # If we didn't finish the grid, we just use what we have
        if not samples:
             # Fallback to random if something went wrong or time was extremely tight
             p = rng.uniform(-5, 5, self.dim)
             fp = func(p)
             return p, fp

        # Sort samples to find the best K starting points
        samples.sort(key=lambda x: x[0])
        
        # Select top K starts for local refinement
        K = 5
        top_k = samples[:K]
        
        # Phase 2: Local Refinement with (1+1)-ES
        used_budget = len(samples)
        remaining_budget = self.budget - used_budget
        
        if remaining_budget > 0 and top_k:
            budget_per_run = max(1, remaining_budget // len(top_k))
            
            for fp, px in top_k:
                if time.time() > deadline:
                    break
                
                x = px.copy()
                fx = fp
                # Initialize sigma based on grid spacing
                # Grid spacing is 10 / (3-1) = 5. 
                # Start with a smaller step to converge, maybe 0.5
                sigma = 1.0
                
                for _ in range(budget_per_run):
                    if time.time() > deadline:
                        break
                    
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.25 # Grow step on success
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.75 # Shrink step on failure
                        # If sigma becomes very small, we've likely converged to local min
                        if sigma < 1e-5:
                            break
                            
        return best_x, best_f
# EVOLVE-BLOCK-END
