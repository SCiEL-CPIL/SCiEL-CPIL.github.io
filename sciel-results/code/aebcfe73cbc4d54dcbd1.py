import numpy as np
import time
from scipy.optimize import minimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
            
            fx = func(x)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            return fx

        # Initial points: Try a few random starts if budget allows, or just one good one.
        # L-BFGS-B is a local optimizer. Bent Cigar is convex (mostly), so one start might be enough if not in a bad local min (there are none for convex).
        # But the "shift" might put the minimum far from 0.
        # Let's try 0 and a random point.
        
        rng = np.random.default_rng(42)
        
        # Start with 0
        x0 = np.zeros(dim)
        
        try:
            res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=[(-5, 5) for _ in range(dim)], options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # If we have budget left, try a random restart
        if calls < budget and time.time() < deadline:
            x1 = rng.uniform(-4, 4, dim)
            try:
                res2 = minimize(wrapped_func, x1, method='L-BFGS-B', bounds=[(-5, 5) for _ in range(dim)], options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_fx
