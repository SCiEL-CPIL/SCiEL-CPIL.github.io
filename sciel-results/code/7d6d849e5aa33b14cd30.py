import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 20 seconds before the 300s limit
        deadline = 280.0 
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start Nelder-Mead
        # Use 10 diverse starting points
        n_starts = 10
        # Reserve some budget for final refinement
        # 10 starts * 40 evals = 400 evals. 100 left.
        evals_per_start = 40 
        
        # Generate starts
        starts = rng.uniform(-5, 5, (n_starts, self.dim))
        
        for start in starts:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Evaluate the start point itself
            f_start = safe_func(start)
            if f_start < best_f:
                best_f = f_start
                best_x = start.copy()
            
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Run Nelder-Mead from this start
            try:
                res = minimize(
                    safe_func, 
                    start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, 
                        'maxfev': evals_per_start,
                        'xatol': 1e-5, 
                        'fatol': 1e-5, 
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 2: Final Refinement
        # If we have budget left and found a candidate, polish it
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            try:
                # Use remaining budget for a tighter polish
                remaining_budget = self.budget - calls_made
                # Cap the polish to avoid wasting too much if it converges quickly
                polish_maxfev = min(remaining_budget, 100)
                
                res = minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, 
                        'maxfev': polish_maxfev,
                        'xatol': 1e-6, 
                        'fatol': 1e-6, 
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Ensure we return something valid even if all fails
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)

        return best_x, best_f
