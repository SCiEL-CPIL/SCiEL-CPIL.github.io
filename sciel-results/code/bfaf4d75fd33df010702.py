import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Try multiple random starts with Powell's method
        n_starts = 10
        for s in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            if calls_made >= budget:
                break

            # Random start
            start = rng.uniform(-5, 5, dim)
            
            # Use Powell's method
            try:
                res = minimize(safe_call, start, method='Powell', 
                               options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8})
            except Exception:
                pass
            
            # Update best if Powell found something better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
            
            # If we have budget left and found a good point, refine further
            if calls_made < budget - 20 and best_f < 1.0:
                try:
                    res2 = minimize(safe_call, best_x, method='Powell', 
                                    options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
