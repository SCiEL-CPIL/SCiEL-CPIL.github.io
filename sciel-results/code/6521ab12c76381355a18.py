import numpy as np
from scipy.optimize import minimize, dual_annealing
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 280.0

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.clip(x, -5.0, 5.0).copy()

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Phase 1: Global Search with Dual Annealing
        # Use a significant portion of the budget
        da_budget = int(budget * 0.7)
        
        try:
            res_da = dual_annealing(wrapped_func, np.zeros(n), 
                                     bounds=bounds,
                                     maxfev=da_budget, 
                                     maxiter=100,
                                     seed=12345)
            update_best(res_da.x, res_da.fun)
        except Exception:
            pass

        # Phase 2: Local Polish with Nelder-Mead
        # Use remaining budget
        remaining = budget - self.call_count
        if remaining > 10:
            try:
                res_nm = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'maxfev': remaining})
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 3: If Dual Annealing was very cheap, do some random restarts with Nelder-Mead
        if self.call_count < budget * 0.5:
            # We have lots of budget left
            # Try 5 random restarts
            rng = np.random.default_rng(42)
            for _ in range(5):
                if self.call_count >= budget - 50:
                    break
                x0 = rng.uniform(-5.0, 5.0, size=n)
                try:
                    res_r = minimize(wrapped_func, x0, method='Nelder-Mead',
                                     options={'maxiter': 50, 'maxfev': 50})
                    update_best(res_r.x, res_r.fun)
                except Exception:
                    pass

        return self.best_x, self.best_f
