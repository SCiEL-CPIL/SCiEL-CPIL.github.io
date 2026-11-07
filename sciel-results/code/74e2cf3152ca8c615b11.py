import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        calls = 0
        best_x = None
        best_f = np.inf
        time_limit = 290  # Safety margin before 300s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: Global Search with Differential Evolution
        # Use about 50% of the budget for global exploration
        global_budget = int(0.5 * self.budget)
        
        # DE parameters tuned for 5D
        popsize = 12
        max_iter_de = max(1, int(global_budget / popsize) - 1)
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds=[(-5, 5)] * self.dim, 
                maxiter=max_iter_de,
                popsize=popsize,
                polish=False,
                tol=1e-10,
                seed=42,
                init='latinhypercube',
                strategy='best1bin'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x, dtype=float).copy()
        except (TimeoutError, Exception):
            pass

        # Phase 2: Local Refinement with CMA-ES (via minimize L-BFGS-B or Nelder-Mead)
        # Since scipy.optimize.minimize doesn't have CMA-ES built-in easily without extra libs,
        # we use Nelder-Mead or L-BFGS-B. L-BFGS-B is good for box constraints.
        # However, for ill-conditioned functions, Nelder-Mead can be robust.
        # Let's try a few local optimizations from the best point found by DE.
        
        remaining_budget = self.budget - calls
        if remaining_budget > 0 and best_x is not None:
            # Try L-BFGS-B
            try:
                res_lm = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                if res_lm.fun < best_f:
                    best_f = res_lm.fun
                    best_x = np.array(res_lm.x, dtype=float).copy()
            except (TimeoutError, Exception):
                pass

            # If we still have budget, do a (1+1)-ES refinement
            remaining_budget = self.budget - calls
            if remaining_budget > 0 and best_x is not None:
                x = best_x.copy()
                fx = best_f
                sigma = 0.1  # Start small
                
                rng = np.random.default_rng(123)
                
                for _ in range(remaining_budget):
                    if time.time() - start_time > time_limit:
                        break
                    
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                    
                    fy = wrapped_func(y)
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    
                    if sigma < 1e-6:
                        break

                if fx < best_f:
                    best_f = fx
                    best_x = x

        return best_x, best_f
