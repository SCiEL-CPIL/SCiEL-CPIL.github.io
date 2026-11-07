import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        bounds = (-5, 5)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counted_func(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            eval_count += 1
            return func(x)

        num_starts = 3
        budget_per_start = self.budget // num_starts
        
        for i in range(num_starts):
            if eval_count >= self.budget:
                break
                
            x0 = rng.uniform(bounds[0], bounds[1], self.dim)
            
            # Estimate maxiter for Nelder-Mead
            # Conservative estimate: dim + 2 evals per iteration
            maxiter = max(1, budget_per_start // (self.dim + 2))
            
            try:
                res = scipy.optimize.minimize(
                    counted_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If exception occurred, check if we updated best_f in the last successful call?
                # No, res is not available if exception.
                # But maybe we can check eval_count? No, we don't track last good point inside scipy.
                # We just rely on the fact that if it crashed, we lost that sub-run's result.
                pass

        # Final polish
        remaining = self.budget - eval_count
        if remaining > 10 and best_x is not None:
            maxiter = max(1, remaining // (self.dim + 2))
            try:
                res = scipy.optimize.minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
