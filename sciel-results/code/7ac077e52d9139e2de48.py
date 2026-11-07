import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
            start_time = time.time()
            call_count = 0
            best_x = None
            best_f = np.inf

            def counted_func(x):
                nonlocal call_count, best_x, best_f
                if call_count >= self.budget:
                    return best_f # Return best known to satisfy API, though ignored
                if time.time() - start_time > 290: # Safety margin
                    return best_f
                call_count += 1
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Phase 1: Global Search with Differential Evolution
            # Budget allocation: ~60% for global, 40% for local
            global_budget = int(self.budget * 0.6)
            
            # DE parameters
            # popsize * (maxiter + 1) < global_budget
            popsize = 10
            # maxiter = (global_budget // popsize) - 1
            maxiter = max(1, (global_budget // popsize) - 1)
            
            try:
                # Bounds are [-5, 5]
                bounds = [(-5, 5)] * self.dim
                res = differential_evolution(
                    counted_func,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-6,
                    seed=42,
                    polish=False # We will do our own polish
                )
                if res.x is not None and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

            # Phase 2: Local Polish with (1+1)-ES
            if best_x is None:
                # Fallback if DE failed
                best_x = np.zeros(self.dim)
                best_f = counted_func(best_x)
            
            remaining_budget = self.budget - call_count
            if remaining_budget > 10:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5 # Start with a moderate step
                rng = np.random.default_rng(42)
                
                for _ in range(remaining_budget):
                    if time.time() - start_time > 295:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = counted_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.85
                best_x, best_f = x, fx

            return best_x, best_f
