import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        num_restarts = 5
        base_budget_per_restart = self.budget // num_restarts
        
        for i in range(num_restarts):
            if time.time() > deadline or self.budget <= 0:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            
            # We will try L-BFGS-B first
            # If it fails or uses too few calls, we might try NM?
            # No, just run L-BFGS-B with a budget cap.
            
            calls_made = 0
            max_calls = min(base_budget_per_restart, self.budget)
            
            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= max_calls:
                    raise Exception("Budget Exhausted")
                calls_made += 1
                self.budget -= 1
                return func(x)

            try:
                max_iter = max(1, max_calls // 6)
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If L-BFGS failed or budget ran out, try Nelder-Mead with remaining budget for this slot?
                # Actually, if budget ran out, we have no more calls for this restart.
                # But if L-BFGS failed due to exception (not budget), we might have calls left.
                # Let's check if we have calls left in this slot.
                remaining_in_slot = max_calls - calls_made
                if remaining_in_slot > 10 and self.budget > 0:
                     # Try NM with the remaining budget
                     calls_made_nm = 0
                     max_calls_nm = remaining_in_slot
                     
                     def wrapped_func_nm(x):
                         nonlocal calls_made_nm
                         if calls_made_nm >= max_calls_nm:
                             raise Exception("Budget Exhausted NM")
                         calls_made_nm += 1
                         self.budget -= 1
                         return func(x)
                     
                     try:
                         res_nm = minimize(wrapped_func_nm, x0, method='Nelder-Mead', options={'maxiter': 1000})
                         if res_nm.fun < best_f:
                             best_f = res_nm.fun
                             best_x = res_nm.x
                     except Exception:
                         pass

        return best_x, best_f
