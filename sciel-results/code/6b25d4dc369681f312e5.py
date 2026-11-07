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
        
        # Track state
        calls = 0
        best_x = None
        best_fx = np.inf
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            
            # Call the actual function
            fx = func(x_clipped)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
                
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
                
            return fx

        # Bent Cigar is convex. Start at 0 as a reasonable initial guess.
        x0 = np.zeros(dim)
        
        # Initialize best with x0
        wrapped_func(x0)
        
        # Phase 1: Powell's method to align search directions with the narrow valley
        # Powell adapts coordinate directions, which is crucial for ill-conditioned
        # functions like Bent Cigar where one dimension has very low curvature.
        try:
            res_powell = minimize(
                wrapped_func, 
                x0, 
                method='Powell',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12, 'maxfev': budget}
            )
            # Update x0 to Powell's result for Phase 2
            x0 = np.clip(res_powell.x, -5.0, 5.0)
        except BudgetExceeded:
            pass
        except Exception:
            pass
        
        # If budget is exhausted, return what we have
        if calls >= budget:
            if best_x is None:
                best_x = x0
                best_fx = func(np.clip(x0, -5.0, 5.0))
            return best_x, best_fx
        
        # Phase 2: L-BFGS-B to polish the final result
        try:
            res_lbgfs = minimize(
                wrapped_func, 
                x0, 
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # If we didn't update best_x (shouldn't happen as we called wrapped_func once), use x0
        if best_x is None:
            best_x = x0
            best_fx = func(np.clip(x0, -5.0, 5.0))

        return best_x, best_fx
