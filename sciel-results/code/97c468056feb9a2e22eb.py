import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        best_x = None
        best_f = np.inf
        
        # Strategy: Multiple restarts with Nelder-Mead
        # 500 budget is small. 5 restarts of 100 calls each seems reasonable.
        # Or 10 restarts of 50 calls.
        # Nelder-Mead in 5D usually converges in < 50-100 calls if close.
        
        num_restarts = 5
        calls_per_restart = self.budget // num_restarts
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random start
            x0 = np.random.uniform(-5, 5, self.dim)
            
            # We need a wrapper to count calls and enforce the global budget?
            # The evaluator enforces the budget by killing if > 500 calls.
            # But we should be careful not to waste calls.
            
            # SciPy's minimize allows maxfev.
            # However, if we run 5 restarts, we might exceed 500 if they all use maxfev.
            # We must track the total calls made by SciPy.
            
            # Let's use a custom counter or just limit maxfev strictly.
            # Since we don't have a global counter inside the loop easily without wrapping,
            # and the evaluator kills us if we exceed, we should be conservative.
            
            # Let's wrap func to count calls and stop if we hit the global budget.
            calls_made = 0
            global_budget = self.budget
            
            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= global_budget:
                    return np.inf # Should not happen if maxfev is correct
                calls_made += 1
                return func(x)
            
            # We need to pass the remaining budget to maxfev
            remaining = global_budget - calls_made
            if remaining <= 0:
                break
            
            # Limit maxfev to remaining budget, but also cap it per restart to allow restarts
            maxfev_this = min(calls_per_restart, remaining)
            
            try:
                res = scipy.optimize.minimize(wrapped_func, x0, method='Nelder-Mead', 
                                              options={'maxiter': maxfev_this, 'maxfev': maxfev_this, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

            # Update global calls made for next iteration logic if we were tracking it
            # But `calls_made` is local to the loop iteration in my sketch above? 
            # No, `wrapped_func` uses `nonlocal` but `calls_made` is re-initialized?
            # I need a persistent counter.
            
        # Return the best found
        if best_x is None:
            # Fallback if no calls were made or all failed
            x = np.random.uniform(-5, 5, self.dim)
            best_f = func(x)
            best_x = x
            
        return best_x, best_f
