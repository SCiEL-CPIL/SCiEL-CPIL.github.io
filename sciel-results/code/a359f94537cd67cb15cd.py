"""Rewritten optimizer using L-BFGS-B for fast local convergence."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        calls_made = 0
        max_calls = self.budget
        
        def time_ok():
            return time.time() < deadline

        def count_func(x):
            nonlocal calls_made
            # Ensure x is a 1D numpy array
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Global Exploration
        # We need to find points in the basin of attraction.
        # Rosenbrock's valley is narrow, so random search is okay but we 
        # should try to be somewhat smart. 
        # Let's do 20 random starts.
        n_starts = 20
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        candidates = []
        
        for i in range(n_starts):
            if not time_ok() or calls_made >= max_calls:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = count_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(self.dim)
            return x, func(x)

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Keep top K candidates for local optimization
        # We want to diversify our local searches but not waste too much budget.
        # Let's take the top 3 distinct points.
        top_k = 3
        start_points = [c[1] for c in candidates[:top_k]]

        # Phase 2: Local Optimization with L-BFGS-B
        # We allocate the remaining budget to these local runs.
        # L-BFGS-B is very efficient, so even a small budget per run can yield
        # significant improvement.
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        # We will run L-BFGS-B from the best candidates.
        # To avoid running out of time, we check time before each run.
        
        for idx, x0 in enumerate(start_points):
            if not time_ok() or calls_made >= max_calls - 10: # Reserve 10 calls for safety
                break
            
            # Define a callback to stop L-BFGS-B if we run out of time or calls
            def callback(xk):
                nonlocal calls_made
                if not time_ok() or calls_made >= max_calls - 5:
                    # Returning True stops the optimization
                    return True
                return False

            try:
                res = minimize(
                    count_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    callback=callback,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If L-BFGS-B fails, we just continue
                pass

        # Phase 3: Final Polish
        # If we have budget left, do one more very precise run from the best point.
        if time_ok() and calls_made < max_calls - 2:
            try:
                # Use a tighter tolerance for the final run
                res_final = minimize(
                    count_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-10}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
