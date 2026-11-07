"""Surrogate-based optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Wrapper to count function calls and enforce deadline
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Ensure x is a valid finite vector
            if not np.all(np.isfinite(x)):
                return np.inf
                
            calls_made += 1
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return val
            except:
                return np.inf

        # Multi-start strategy
        # We will run several short Nelder-Mead optimizations
        # Each run uses a portion of the budget
        
        num_restarts = 5
        calls_per_run = max(10, budget // num_restarts)
        
        # Initial best point: center
        current_best = np.zeros(dim)
        current_best_f = wrapped_func(current_best)
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = current_best.copy()
            
        for i in range(num_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Choose starting point
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                # Random start
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb best found so far
                noise = rng.normal(0, 1.0, dim)
                x0 = best_x + noise
                # Clip to box
                x0 = np.clip(x0, -5, 5)
                
            # Run Nelder-Mead
            # Set maxfev to remaining budget or calls_per_run
            maxfev = min(budget - calls_made, calls_per_run)
            if maxfev <= 0:
                break
            
            try:
                result = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': maxfev,
                        'xatol': 1e-6,
                        'fatol': 1e-8
                    }
                )
                
                if result.success or result.fun < best_f:
                    if np.all(np.isfinite(result.x)) and result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
            except Exception:
                pass
                
            # Check time
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
