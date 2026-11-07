import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 300s limit, use 290s for safety
        self.deadline = self.start_time + 290

    def __call__(self, func):
        rng = np.random.default_rng(42)
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        def counted_func(x):
            nonlocal eval_count
            if time.time() > self.deadline:
                raise TimeoutError
            eval_count += 1
            if eval_count > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)

        # Phase 1: Global Search
        # Use 50 evaluations for global search
        n_global = 50
        for i in range(n_global):
            if time.time() > self.deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            try:
                fx = counted_func(x)
            except Exception:
                continue
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            return np.zeros(self.dim), np.inf

        # Phase 2: Local Refinement with L-BFGS-B
        remaining_budget = self.budget - eval_count
        if remaining_budget > 10:
            bounds = [(-5.0, 5.0)] * self.dim
            try:
                res = scipy.optimize.minimize(
                    counted_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxfun': remaining_budget, 'maxiter': 100}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
