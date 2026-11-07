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

        # Initial guess
        # Bent Cigar is convex. 0 is a reasonable start.
        # If the shift is large, Powell will still converge, but might take more iterations.
        # Given the convexity, any start works.
        x0 = np.zeros(dim)
        
        # Evaluate x0 to initialize best
        wrapped_func(x0)
        
        try:
            # Use Powell's method
            # Powell is derivative-free and handles ill-conditioned problems well
            # by updating its search directions to align with the principal axes.
            res = minimize(
                wrapped_func, 
                x0, 
                method='Powell',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
        except BudgetExceeded:
            pass
        except Exception:
            # Catch any other errors
            pass

        # If we have significant budget left, we could try another method or restart
        # But Powell should be robust.
        
        if best_x is None:
            best_x = x0
            best_fx = func(np.clip(x0, -5.0, 5.0))

        return best_x, best_fx
