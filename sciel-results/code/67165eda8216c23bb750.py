import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        best_x = None
        best_f = float('inf')
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal self
            if self.calls >= self.budget:
                return 1e10 
            self.calls += 1
            # Check time limit (safety margin)
            if time.time() - self.start_time > 290: # 290s limit to be safe
                return 1e10
            return func(x)
            
        rng = np.random.default_rng(0)
        
        # Strategy:
        # 1. Global search with Differential Evolution (derivative-free, handles multimodality)
        # 2. Local refinement with L-BFGS-B (fast convergence on smooth basins)
        
        # Phase 1: Differential Evolution
        # popsize=10, maxiter=20 -> ~200 calls
        # This leaves ~100 calls for Phase 2.
        
        try:
            # bounds are (-5, 5) for all dims
            bounds = [(-5, 5)] * self.dim
            
            # Use differential_evolution for global search
            # We set polish=False to control the local search budget manually
            de_res = differential_evolution(
                wrapped_func, 
                bounds, 
                popsize=10, 
                maxiter=20, 
                init='latinhypercube',
                polish=False,
                tol=0.01,
                seed=0
            )
            
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x
                
        except Exception:
            # Fallback if DE fails
            pass

        # Phase 2: Local Refinement with L-BFGS-B
        # Use remaining budget
        remaining_budget = self.budget - self.calls
        
        if best_x is None:
            # If DE didn't find anything or failed, start from random
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = wrapped_func(best_x)
            remaining_budget -= 1

        if remaining_budget > 10:
            try:
                # L-BFGS-B is efficient for the final polish
                # maxfun should not exceed remaining budget
                lbfgs_res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim, 
                    options={'maxfun': remaining_budget, 'maxiter': remaining_budget}
                )
                
                if lbfgs_res.fun < best_f:
                    best_f = lbfgs_res.fun
                    best_x = lbfgs_res.x
                    
            except Exception:
                pass
        
        # Ensure at least one call was made
        if self.calls == 0:
             x_test = rng.uniform(-5, 5, self.dim)
             best_f = wrapped_func(x_test)
             best_x = x_test

        return best_x, best_f
