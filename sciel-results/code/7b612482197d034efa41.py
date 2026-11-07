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
        start_time = time.time()
        deadline = start_time + 280.0

        # State variables
        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Use a fixed random seed for reproducibility and to mimic the successful inspiration
        rng = np.random.default_rng(0)

        def safe_func(x):
            nonlocal calls_made, best_x, best_fx
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES Global Search
        # Allocate 80% of the budget to the evolutionary step
        es_budget = int(budget * 0.8)
        
        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = safe_func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        
        # (1+1)-ES loop
        while calls_made < es_budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = safe_func(y)
            
            # Update
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5  # Expand step on success
            else:
                sigma *= 0.87 # Shrink step on failure
            
            # Safety check for sigma
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 10.0:
                sigma = 10.0

        # Phase 2: Local Refinement
        # Use remaining budget for local search
        remaining_budget = budget - calls_made
        if remaining_budget > 5 and best_x is not None and time.time() < deadline:
            # Use Powell method for derivative-free local optimization
            # We wrap the function to ensure we don't exceed the total budget
            # scipy minimize might make multiple calls internally.
            # We need to be careful with the budget.
            
            # Create a wrapper that counts calls
            local_calls = 0
            def local_func(x):
                nonlocal local_calls, calls_made
                if local_calls >= remaining_budget:
                    return best_fx # Return a large value or current best to stop improvement
                local_calls += 1
                return safe_func(x)
            
            try:
                # Powell is robust for smooth functions
                res = minimize(
                    local_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': remaining_budget, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                # Update best if improvement found
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
