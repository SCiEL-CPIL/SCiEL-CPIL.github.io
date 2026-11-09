"""Rewritten optimizer using Multi-Start Nelder-Mead for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Nelder-Mead with budget-aware restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Number of restarts
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        for i in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate random start
            # Use a wider range for the first few, or just uniform
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Define a wrapper to count function calls and enforce budget
            calls = [0]
            limit = budget_per_restart
            
            def constrained_func(x):
                if time.time() > deadline:
                    return np.inf
                if calls[0] >= limit:
                    return np.inf
                calls[0] += 1
                try:
                    fx = func(x)
                    return fx
                except Exception:
                    return np.inf

            # Run Nelder-Mead
            # initial_simplex_size: controls the size of the initial simplex
            # For Rosenbrock, a smallish size is often better to stay in the valley
            # but too small might get stuck. 0.1 to 0.5 is typical.
            try:
                res = minimize(
                    constrained_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': limit * 2, # Max iterations, actual calls are tracked
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass

        # If no valid best_x found (shouldn't happen), return a random point
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
