import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.budget = self.budget  # reset budget for this instance
        
        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            self.calls += 1
            return func(x)

        best_x = None
        best_f = np.inf
        
        # Strategy: Multi-start Powell
        # Use 4 restarts.
        n_restarts = 4
        calls_per_restart = self.budget // n_restarts
        
        rng = np.random.default_rng(0)
        
        for i in range(n_restarts):
            if self.calls >= self.budget:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                # Powell is effective for anisotropic functions
                res = minimize(wrapped_func, x0, method='Powell', 
                               options={'maxfev': calls_per_restart, 'xtol': 1e-10, 'ftol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Final polish with remaining budget
        if best_x is not None and self.calls < self.budget:
            remaining = self.budget - self.calls
            if remaining > 5:
                try:
                    res = minimize(wrapped_func, best_x, method='Powell',
                                   options={'maxfev': remaining, 'xtol': 1e-12, 'ftol': 1e-12})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
