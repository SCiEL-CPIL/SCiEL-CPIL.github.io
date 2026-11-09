"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to evaluate and update best
        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            # Check limits before calling func
            if calls_made >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Clip to bounds
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            
            # Update best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Sampling
        # Use 40 initial points to cover the space
        # This provides a good initial estimate of the landscape
        n_initial = 40
        rng = np.random.default_rng(42)
        initial_points = rng.uniform(-5, 5, (n_initial, dim))
        
        initial_results = []
        for x in initial_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = safe_eval(x)
            if np.isfinite(fx):
                initial_results.append((fx, x))
        
        if not initial_results:
            # Fallback if something went wrong (e.g., immediate timeout)
            x0 = np.zeros(dim)
            safe_eval(x0)
            return best_x, best_f

        # Sort by fitness to prioritize best basins
        initial_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # Run NM from the top initial points
        # Budget remaining is ~460. Each NM run might use 50-150 calls.
        # We can run ~3-5 instances.
        
        max_refinements = 6
        
        for i in range(min(max_refinements, len(initial_results))):
            if calls_made >= budget or time.time() > deadline:
                break
            
            fx0, x0 = initial_results[i]
            
            # Run Nelder-Mead
            # adaptive=True helps with step size adjustment
            res = minimize(safe_eval, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            # safe_eval has already updated best_x and best_f
            # But we can double check if res.fun is better (it should be, as it's the min of the run)
            if np.isfinite(res.fun) and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 3: Additional Restarts if budget remains
        # If we still have > 50 calls left, try 5 more random points and refine the best one
        if calls_made < budget - 50 and time.time() < deadline:
            n_extra = 10
            extra_points = rng.uniform(-5, 5, (n_extra, dim))
            extra_results = []
            for x in extra_points:
                if calls_made >= budget or time.time() > deadline:
                    break
                fx = safe_eval(x)
                if np.isfinite(fx):
                    extra_results.append((fx, x))
            
            if extra_results:
                extra_results.sort(key=lambda t: t[0])
                # Refine the best extra point
                fx0, x0 = extra_results[0]
                # Use fewer iterations as we are likely near the end
                res = minimize(safe_eval, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
