import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        call_count = 0
        best_x = None
        best_f = float('inf')

        def count_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Quick random search to find a good starting point
        n_init = 10
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, dim)
            count_call(x_try)
        
        if best_x is None:
            best_x = np.zeros(dim)
            count_call(best_x)

        # Phase 2: CMA-ES refinement
        remaining_budget = budget - call_count
        if remaining_budget > 10 and time.time() < deadline:
            # Estimate max iterations for CMA-ES
            # Lambda ~ 9 for dim=5
            max_iter = max(10, remaining_budget // 10)
            
            try:
                res = minimize(count_call, best_x, method='CMA',
                               options={'maxiter': max_iter, 
                                        'xatol': 1e-4, 
                                        'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
