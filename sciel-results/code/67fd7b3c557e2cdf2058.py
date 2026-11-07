import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0 # 10s safety margin
        
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() - start_time > timeout:
                raise TimeoutError()
            x = np.clip(np.asarray(x, dtype=float), -self.bound, self.bound)
            self.calls += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Global Search
        # Use a small number of random restarts to find a good basin
        # 5D space is small, 30-50 points is usually enough to find the valley
        n_global = 50
        rng = np.random.default_rng(42)
        
        # Generate random points
        # We can also include 0 and some structured points
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-self.bound, self.bound, self.dim)
            candidates.append(x)
        
        # Evaluate candidates
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            wrapped_func(x)
            
        if self.best_x is None:
            # Fallback if no calls made (should not happen)
            x0 = np.zeros(self.dim)
            wrapped_func(x0)
            
        # Phase 2: Local Optimization
        # Use L-BFGS-B which is robust and uses finite differences
        # It is very effective for smooth functions like Rosenbrock
        remaining_budget = self.budget - self.calls
        
        if remaining_budget > 0:
            bounds = [(-self.bound, self.bound)] * self.dim
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': remaining_budget,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20 # Limit line search steps to prevent wasting budget on one step
                    }
                )
            except Exception:
                pass

            # Phase 3: Polish with Nelder-Mead if budget remains and L-BFGS-B didn't converge to machine precision
            # Nelder-Mead is derivative-free and can sometimes escape shallow local minima or refine further
            remaining_budget = self.budget - self.calls
            if remaining_budget > 10:
                try:
                    res_nm = scipy.optimize.minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='Nelder-Mead', 
                        bounds=bounds, # Note: Nelder-Mead in scipy 1.7+ supports bounds, but older versions don't. 
                                       # Since we clip in wrapped_func, bounds are effectively handled.
                        options={
                            'maxiter': remaining_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-12
                        }
                    )
                except Exception:
                    pass

        return self.best_x, self.best_f
