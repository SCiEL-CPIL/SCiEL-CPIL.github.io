import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        # Initial guess
        x0 = np.zeros(dim)
        best_x = x0.copy()
        best_f = func(x0)
        calls_used = 1

        # Use Nelder-Mead for robust optimization
        # It handles both linear and non-linear functions well.
        # We set maxiter to a high number, but we wrap func to track calls and time.
        
        # We need to reserve some calls for final verification or just let it run.
        # Nelder-Mead in scipy uses a specific number of function evaluations per iteration.
        # We will monitor the time and calls.

        def wrapped_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_used += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Options for Nelder-Mead
        # maxiter: maximum number of iterations.
        # xatol, fatol: tolerances.
        # We want to use most of the budget.
        # Each iteration of Nelder-Mead in d dimensions uses ~d+1 function evaluations.
        # So maxiter * (d+1) <= budget.
        # 500 / 6 = 83 iterations.
        # Let's set maxiter to 100 to be safe, and rely on the wrapper to stop.
        
        try:
            res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        except Exception:
            pass

        # If we have budget left, or if Nelder-Mead stopped early,
        # we can try a few more points or just re-evaluate best.
        # For linear functions, if Nelder-Mead is near a vertex, it might oscillate.
        # We can check the best_x and see if it's close to a vertex.
        # If so, evaluate the exact vertex.
        
        # Check if best_x is close to a vertex
        # For each dimension, if |x_i| > 4.5, it's likely at the boundary.
        # Let's just re-evaluate the best point found.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            f_recheck = func(best_x)
            calls_used += 1
            if f_recheck < best_f:
                best_f = f_recheck

        return best_x, best_f
