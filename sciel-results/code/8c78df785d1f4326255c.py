import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        
        rng = np.random.default_rng(0)
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def check_time():
            return time.time() > deadline

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget or check_time():
                raise Exception("Budget or time exceeded")
            calls_made += 1
            return func(x)

        # Phase 1: Initial Random Sampling
        # Sample a few points to find a good starting basin
        n_initial = 5
        for _ in range(n_initial):
            if calls_made >= self.budget or check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Iterative Local Refinement
        # Use Nelder-Mead to refine the best point found
        # Then restart from new random points if budget allows
        
        while calls_made < self.budget and not check_time():
            # Run Nelder-Mead from current best
            # Limit iterations to avoid using too much budget in one go if it doesn't converge
            # But Nelder-Mead is fast. Let it run until convergence or budget low.
            
            # We need to be careful: minimize() will call safe_func.
            # If safe_func raises an exception, minimize will crash.
            # So we must limit maxiter such that we don't exceed budget.
            
            remaining = self.budget - calls_made
            if remaining < 10: # Need some calls for Nelder-Mead
                break
                
            # Estimate maxiter: each iteration uses ~dim+1 calls
            # Leave a buffer
            max_iter = max(1, (remaining - 5) // (self.dim + 1))
            
            try:
                res = minimize(safe_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If minimize fails (e.g. due to budget check in safe_func), 
                # we might have already updated best_x/best_f in partial steps?
                # No, minimize returns at the end. If safe_func raises, it propagates.
                # So we must catch it.
                pass

            # If we still have budget, try a new random restart
            if calls_made < self.budget and not check_time():
                x_new = rng.uniform(-5, 5, self.dim)
                # Evaluate x_new
                try:
                    fx_new = safe_func(x_new)
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                except Exception:
                    break

        return best_x, best_f
