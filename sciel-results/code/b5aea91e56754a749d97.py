"""Rewritten optimizer for BBOB-style f14 (Different powers) task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer.
    Uses scipy's L-BFGS-B with multiple random initializations to handle
    non-convexity and rotations. This is a gradient-based local search
    strategy, distinct from evolutionary methods like CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0
        bounds = [(-5.0, 5.0)] * dim

        # Helper to wrap func and count calls
        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                # Return a dummy value to stop minimize, but don't call func
                # Returning best_f might cause it to think it converged
                return best_f 
            calls_used += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # 1. Initial random scan to find a decent starting point
        # Use 10% of budget or at least 10 calls for random search
        num_random = max(10, budget // 10)
        for _ in range(num_random):
            if calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)
            
        # 2. Multi-start L-BFGS-B
        # We will try up to 'max_starts' initial points.
        # Each L-BFGS-B run has a max_iter cap to prevent getting stuck.
        # We leave some budget for final random exploration if needed.
        
        max_starts = 20
        # We want to ensure we don't exhaust budget in the first start
        # If budget is 500, and we used 50 for random, we have 450 left.
        # 450 / 20 starts = 22 calls per start.
        # maxiter=20 is a good fit.
        
        for i in range(max_starts):
            if calls_used > budget - 5: # Leave a small safety margin
                break
                
            # Generate random start
            # We can use the current best_x or a new random point.
            # Using new random points ensures diversity.
            x0 = rng.uniform(-5, 5, dim)
            
            # Run L-BFGS-B
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=bounds, 
                               options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                pass
                
            # If we still have budget, we can try another start
            if calls_used >= budget:
                break

        # 3. Final random exploration if budget remains
        # This helps if L-BFGS-B got stuck in a local minimum and we missed the global one
        while calls_used < budget:
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
