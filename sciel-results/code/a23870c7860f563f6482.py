import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start Nelder-Mead
        n_starts = 6
        budget_per_start = budget // n_starts
        
        rng = np.random.default_rng(42)
        starts = [rng.uniform(-5, 5, d) for _ in range(n_starts)]
        
        for x0 in starts:
            if calls >= budget:
                break
            if time.time() - start_time > 250:
                break
                
            # Nelder-Mead uses d+1 function calls per iteration roughly
            max_iter = max(1, budget_per_start // (d + 1))
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 
                                        'xatol': 1e-4, 
                                        'fatol': 1e-4,
                                        'adaptive': True})
            except Exception:
                continue
                
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # If no calls were made (should not happen), do one
        if calls == 0:
             x0 = np.zeros(d)
             wrapped_func(x0)

        return best_x, best_f
