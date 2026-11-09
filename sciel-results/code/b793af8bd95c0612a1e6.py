import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        calls_made = 0
        max_calls = self.budget
        dim = self.dim
        
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return 1e10
            calls_made += 1
            if time.time() > deadline:
                return 1e10
            # Ensure x is a finite vector
            if not np.all(np.isfinite(x)):
                return 1e10
            return func(x)

        best_x = np.zeros(dim)
        best_f = float('inf')
        candidates = []
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # 10 starts, low iteration count
        n_starts = 10
        maxiter_phase1 = 5
        
        for i in range(n_starts):
            if calls_made >= max_calls or time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter_phase1, 'xatol': 1e-2, 'fatol': 1e-2})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                candidates.append((res.fun, res.x))
            except Exception:
                pass

        if not candidates:
            # Fallback
            x0 = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x0)
            return x0, fx

        # Sort candidates
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Refine top candidates
        remaining_calls = max_calls - calls_made
        if remaining_calls > 20:
            top_k = 3
            calls_per_refine = remaining_calls // (top_k + 1) # +1 for Phase 3
            if calls_per_refine > 5:
                for i in range(min(top_k, len(candidates))):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    f_curr, x_curr = candidates[i]
                    try:
                        res = minimize(wrapped_func, x_curr, method='Nelder-Mead', 
                                       options={'maxiter': calls_per_refine, 'xatol': 1e-3, 'fatol': 1e-3})
                        candidates.append((res.fun, res.x))
                    except Exception:
                        pass
        
        # Update best
        candidates.sort(key=lambda x: x[0])
        best_f, best_x = candidates[0]
        
        # Phase 3: Final (1+1)-ES
        remaining_calls = max_calls - calls_made
        if remaining_calls > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            best_x, best_f = x, fx

        return best_x, best_f
