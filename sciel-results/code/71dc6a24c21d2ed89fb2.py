import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start Nelder-Mead
        num_starts = 5
        # Allocate 80% of budget to multi-start
        phase1_budget = int(total_budget * 0.8)
        budget_per_start = phase1_budget // num_starts
        
        for s in range(num_starts):
            if time.time() > deadline:
                break
            if phase1_budget <= 0:
                break
                
            x0 = rng.uniform(-5, 5, n)
            
            calls = [0]
            def wrapped_func(x):
                nonlocal calls
                calls[0] += 1
                if calls[0] > budget_per_start:
                    return best_f
                return func(np.clip(x, -5, 5))
                
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': budget_per_start, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
            phase1_budget -= calls[0]

        # Phase 2: Polish the best point with remaining budget
        if best_x is not None and time.time() < deadline and total_budget > 0:
            remaining_budget = total_budget - (phase1_budget + (num_starts * budget_per_start - phase1_budget)) # This calculation is messy
            # Simpler: just check time and remaining calls
            # We don't track total calls globally easily in this structure.
            # Let's just do one more Nelder-Mead on the best point with a small budget.
            
            if time.time() < deadline - 10:
                calls2 = [0]
                max_iter2 = 50 # Small polish
                def wrapped_func2(x):
                    nonlocal calls2
                    calls2[0] += 1
                    return func(np.clip(x, -5, 5))
                
                try:
                    res2 = minimize(wrapped_func2, best_x, method='Nelder-Mead', 
                                   options={'maxiter': max_iter2, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x
                except Exception:
                    pass

        return best_x, best_f
