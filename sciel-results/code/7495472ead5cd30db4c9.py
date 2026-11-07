import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        max_time = 310.0
        
        def time_ok():
            return time.time() - start_time < max_time

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return best_f
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Sampling
        # Use a Halton sequence for better space-filling than random
        n_initial = 20
        if budget < n_initial + 10:
            # If budget is very small, just do random
            points = np.random.uniform(-5, 5, (n_initial, dim))
        else:
            # Halton sequence
            sampler = qmc.Halton(d=dim, scramble=True)
            # Generate points in [0, 1]^dim
            halton_points = sampler.random(n_initial)
            # Scale to [-5, 5]^dim
            points = halton_points * 10 - 5

        # Evaluate initial points
        for p in points:
            if not time_ok():
                break
            wrapped_func(p)

        # Sort initial points by fitness
        # We need to retrieve the fitness values. 
        # Since wrapped_func doesn't return a list, we should store them.
        # Let's refactor to store fitnesses.
        
        # Re-doing Phase 1 to store fitnesses properly
        calls = 0
        best_x = None
        best_f = np.inf
        initial_results = []

        def wrapped_func_track(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return best_f
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Generate points again
        if budget < 30:
             points = np.random.uniform(-5, 5, (min(10, budget), dim))
        else:
             n_initial = 20
             sampler = qmc.Halton(d=dim, scramble=True)
             halton_points = sampler.random(n_initial)
             points = halton_points * 10 - 5

        for p in points:
            if not time_ok():
                break
            f_val = wrapped_func_track(p)
            initial_results.append((f_val, p.copy()))

        # Sort by fitness
        initial_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement
        # Take top K candidates
        # Budget remaining
        remaining = budget - calls
        if remaining > 0 and time_ok():
            # Determine how many local searches to run
            # Nelder-Mead in 5D takes roughly 5*dim + 1 initial simplex, then iterations.
            # Typically < 100 calls for good convergence.
            # Let's allocate remaining budget to top candidates.
            
            # If we have a lot of budget, we can run more.
            # Let's try top 5 candidates.
            num_refine = min(5, len(initial_results))
            budget_per_refine = max(10, remaining // num_refine)
            
            for i in range(num_refine):
                if not time_ok():
                    break
                if calls >= budget:
                    break
                
                f_init, x_init = initial_results[i]
                
                # Run Nelder-Mead
                # maxfev controls the number of function evaluations
                # We want to use up to budget_per_refine, but not exceed total budget
                maxfev = min(budget_per_refine, budget - calls)
                
                if maxfev < 10:
                    continue

                try:
                    res = minimize(
                        wrapped_func_track, 
                        x_init, 
                        method='Nelder-Mead', 
                        options={'maxfev': maxfev, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                except Exception:
                    continue

                # Update best if res is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

        return best_x, best_f
