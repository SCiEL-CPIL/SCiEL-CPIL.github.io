"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Explicit deadline guard
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Global Scattering
        # We spend a fraction of the budget to find good starting points.
        # 50 points is cheap and helps escape bad local minima or wide valleys.
        n_scatter = min(50, self.budget // 4) 
        points = []
        fvals = []
        
        for _ in range(n_scatter):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            points.append(x)
            fvals.append(fx)
            
        if not points:
             # Fallback if even scattering fails (shouldn't happen)
             x = np.zeros(self.dim)
             return x, func(x)
             
        # Sort by fitness to prioritize best seeds
        indices = np.argsort(fvals)
        points = [points[i] for i in indices]
        fvals = [fvals[i] for i in indices]
        
        best_x = points[0]
        best_f = fvals[0]
        
        # Phase 2: Local Refinement
        # Use the remaining budget for Nelder-Mead on the top seeds.
        # Nelder-Mead is derivative-free and robust for Rosenbrock.
        # We limit the number of function calls per seed to ensure we don't
        # blow the total budget.
        
        # Reserve some calls for safety / potential extra polish if time permits
        remaining_budget = self.budget - len(points)
        
        # How many seeds do we refine?
        # Let's try to refine the top 3-5 seeds if budget allows.
        # Each Nelder-Mead run will consume up to 'max_calls_per_seed' evaluations.
        
        # Estimate max calls per seed. 
        # If we refine 5 seeds, and we have ~450 calls left, that's ~90 calls per seed.
        # Nelder-Mead in 5D often converges in < 100 calls for Rosenbrock if start is close.
        
        n_seeds_to_refine = 5
        # Ensure we don't try to refine more seeds than we have
        n_seeds_to_refine = min(n_seeds_to_refine, len(points))
        
        # Dynamic max calls per seed based on remaining budget
        # Leave a small buffer for the time check overhead
        max_calls_per_seed = max(10, (remaining_budget // n_seeds_to_refine) - 5)
        
        # Define a wrapper to count calls and enforce limits/time
        class CallCounter:
            def __init__(self, limit):
                self.limit = limit
                self.count = 0
            def __call__(self, x):
                if self.count >= self.limit:
                    # Return a very large number to stop optimization
                    return 1e10
                self.count += 1
                return func(x)
        
        for i in range(n_seeds_to_refine):
            if time.time() > deadline:
                break
                
            start_x = points[i]
            
            # Create a counter for this specific run
            counter = CallCounter(max_calls_per_seed)
            
            # Wrap func to use counter
            def wrapped_func(x, counter=counter):
                return counter(x)
                
            try:
                # Nelder-Mead with bounds handling (it doesn't have native bounds, 
                # but Rosenbrock penalty grows outside, and our search space is small.
                # We can clip x inside the wrapper if needed, but Nelder-Mead usually
                # stays within reasonable bounds if initialized well. 
                # To be safe, we clip in the wrapper.)
                
                def safe_func(x, counter=counter):
                    x_clipped = np.clip(x, -5, 5)
                    return counter(x_clipped)
                    
                res = minimize(
                    safe_func, 
                    start_x, 
                    method='Nelder-Mead',
                    options={'maxiter': max_calls_per_seed, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                # Check result
                if res.success or res.fun < best_f:
                    # res.fun is the value at res.x
                    # Note: safe_func clips, so res.x might be slightly out, 
                    # but res.fun is valid. We should update best_x to clipped res.x.
                    candidate_x = np.clip(res.x, -5, 5)
                    # We trust res.fun as it was calculated on the clipped path
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = candidate_x
                        
            except Exception:
                # If minimizer fails, skip
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
