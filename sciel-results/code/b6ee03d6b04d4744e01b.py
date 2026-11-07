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
        time_limit = 285  # Safety margin before 300s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            
            calls += 1
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: Global Search with Differential Evolution
        # Use about 60% of the budget for global exploration
        global_budget = int(0.6 * self.budget)
        
        # DE parameters tuned for 5D
        popsize = 10
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

        # Phase 2: Multi-start local refinement
        remaining_budget = self.budget - calls
        
        if remaining_budget > 0 and best_x is not None:
            # Collect candidate starting points
            candidates = [best_x.copy()]
            
            # Add a few perturbed versions of the best point
            rng = np.random.default_rng(123)
            for i in range(min(3, remaining_budget // 50)):
                perturbation = 0.5 * rng.standard_normal(self.dim)
                cand = np.clip(best_x + perturbation, -5, 5)
                candidates.append(cand)
            
            # Try local optimization from each candidate
            for cand in candidates:
                if calls >= self.budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                # Try L-BFGS-B
                try:
                    res_lm = minimize(
                        wrapped_func,
                        cand,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-7}
                    )
                    if res_lm.fun < best_f:
                        best_f = res_lm.fun
                        best_x = np.array(res_lm.x, dtype=float).copy()
                except (TimeoutError, Exception):
                    pass
                
                # Follow with (1+1)-ES refinement
                remaining_budget = self.budget - calls
                if remaining_budget <= 0 or time.time() - start_time > time_limit:
                    break
                
                if best_x is not None:
                    x = best_x.copy()
                    fx = best_f
                    sigma = 0.05
                    
                    es_rng = np.random.default_rng(456)
                    
                    for _ in range(min(remaining_budget, 100)):
                        if time.time() - start_time > time_limit:
                            break
                        if calls >= self.budget:
                            break
                        
                        y = x + sigma * es_rng.standard_normal(self.dim)
                        y = np.clip(y, -5, 5)
                        
                        fy = wrapped_func(y)
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma *= 1.05
                        else:
                            sigma *= 0.95
                        
                        if sigma < 1e-7:
                            break

                    if fx < best_f:
                        best_f = fx
                        best_x = x

        return best_x, best_f
