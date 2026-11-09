import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        
        # 1. Deterministic Sobol sampling for initial exploration
        # Use 50 points for good coverage
        n_sobol = min(50, budget // 2)
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=n, scramble=True)
        # We need to skip the first point (0,0,...,0) as it's not in the interior usually, 
        # or just map [0,1] to [-5,5].
        sobol_points = sampler.random(n_sobol)
        # Map [0, 1] to [-5, 5]
        sobol_points = -5 + 10 * sobol_points
        
        best_x = sobol_points[0].copy()
        best_f = func(best_x)
        count = 1
        
        for i in range(1, n_sobol):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # 2. Local refinement with Nelder-Mead
        # Remaining budget
        remaining_budget = budget - count
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
            
        # Wrapper to track calls and enforce budget/time
        call_count = [0]
        
        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > remaining_budget or time.time() > deadline:
                # Return a large value to discourage further exploration in this direction
                # or raise an exception. Returning large value is safer for scipy.
                return 1e100
            return func(x)
        
        try:
            # Nelder-Mead options
            # maxiter: max number of iterations. Each iter is ~n+1 evals.
            # maxfev: max number of function evaluations.
            options = {
                'maxiter': 100,
                'maxfev': remaining_budget,
                'xatol': 1e-6,
                'fatol': 1e-8
            }
            
            result = minimize(wrapped_func, best_x, method='Nelder-Mead', options=options)
            
            # Update best if result is better
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception:
            # If minimize crashes, we still have the best_x from Sobol
            pass
            
        return best_x, best_f
