import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget
        
        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > total_budget:
                raise ValueError("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        bounds = [(-5.0, 5.0)] * n
        
        # Strategy: Multi-start L-BFGS-B
        # 1. Use 20% of budget for global exploration (random points + short L-BFGS-B)
        # 2. Use 80% of budget for local refinement from the best point
        
        n_starts = 10
        per_start_budget = max(10, int(total_budget * 0.05))
        
        for i in range(n_starts):
            if time.time() > deadline or calls_made[0] >= total_budget:
                break
            
            # Random start
            x0 = np.random.uniform(-4.0, 4.0, n)
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': per_start_budget, 'maxfun': per_start_budget}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Final refinement
        if time.time() < deadline - 5.0 and calls_made[0] < total_budget - 10:
            try:
                res_final = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = np.clip(res_final.x, -5.0, 5.0)
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return best_x, best_f
