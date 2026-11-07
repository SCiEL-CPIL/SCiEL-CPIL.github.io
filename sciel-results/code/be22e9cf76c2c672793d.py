import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        calls_made = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")
            calls_made += 1
            return func(x)

        # Phase 1: Global Search with Differential Evolution
        # Target ~400 calls
        # popsize=10, maxiter=39 -> 10*(1+39) = 400 calls
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        try:
            res_de = differential_evolution(
                counted_func,
                bounds,
                maxiter=39,
                popsize=10,
                tol=1e-9,
                atol=1e-12,
                init='sobol',
                seed=0,
                polish=False
            )
            best_x = res_de.x
            best_f = res_de.fun
        except Exception:
            # If DE fails, try a few random starts with NM
            rng = np.random.default_rng(0)
            for _ in range(5):
                if calls_made >= self.budget: break
                x0 = rng.uniform(-5, 5, self.dim)
                try:
                    res = minimize(counted_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    break

        # Phase 2: Local Polish with Nelder-Mead
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            remaining = self.budget - calls_made
            # NM calls approx dim * maxiter
            maxiter_nm = max(1, remaining // (self.dim + 1))
            
            try:
                res_nm = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter_nm, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        # Final safety check: if no valid best_x, do one random call
        if best_x is None:
            x0 = np.zeros(self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
