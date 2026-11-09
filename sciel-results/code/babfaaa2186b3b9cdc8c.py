import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * n
        
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        # Deadline guard
        start_time = time.time()
        deadline = 290.0  # Safety margin

        # Multi-start strategy
        # We will use several random starting points and run L-BFGS-B from each.
        # L-BFGS-B is efficient for smooth functions like Rosenbrock.
        
        # Number of restarts
        n_restarts = 10
        
        # Pre-generate random points to ensure diversity
        # Use a fixed seed? No, the task says "hidden seeded shift", but we can use our own randomness.
        # However, deterministic behavior is often safer for reproducibility if not required.
        # The task doesn't forbid randomness.
        
        rng = np.random.default_rng(42) # Fixed seed for consistency in our internal logic? 
        # Actually, let's just use standard np.random or a new instance.
        
        for r in range(n_restarts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize start point
            # For r=0, start at 0. For others, random.
            if r == 0:
                x0 = np.zeros(n)
            else:
                # Sample from the box
                x0 = rng.uniform(-5.0, 5.0, size=n)
            
            # We need to wrap func to count calls and check time
            # But minimize() calls func many times.
            # We can pass a wrapper.
            
            # Check if we have enough budget for at least one step
            if calls >= budget:
                break
                
            # Run L-BFGS-B
            # maxiter: limit iterations to prevent runaway
            # maxfun: limit function evaluations
            # We want to use the remaining budget.
            # However, minimize doesn't stop exactly at budget if we set maxfun high.
            # We'll set maxfun to a reasonable chunk, say 50, or the remaining budget.
            
            remaining = budget - calls
            if remaining <= 0:
                break
            
            # Limit maxfun to remaining budget, but also cap it to avoid long runs if convergence is slow
            # L-BFGS-B is fast, usually converges in < 50 evals for 5D Rosenbrock if close.
            # If far, it might take more.
            maxfun = min(remaining, 100) 
            maxiter = 100
            
            # Wrapper to track calls
            # Note: minimize might call func with a vector.
            
            # We need a closure that updates calls and best_f
            # And checks time.
            
            # Define a local function for the optimizer
            def func_wrapper(x):
                nonlocal calls, best_f, best_x
                if calls >= budget or time.time() > deadline:
                    # Return a large value to stop? Or raise?
                    # Raising might crash minimize. Returning inf is safer.
                    return float('inf')
                
                # Ensure x is a valid numpy array of length n
                # func clips to box, so we can pass x directly.
                # But we should ensure x is 1D float array.
                x = np.asarray(x, dtype=float)
                if x.shape != (n,):
                    x = x.reshape(n)
                
                calls += 1
                f_val = func(x)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                    
                return f_val
            
            try:
                res = minimize(func_wrapper, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxfun': maxfun, 'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                # If minimize crashes, continue to next restart
                pass
                
            # After minimize, check if we broke out early
            if calls >= budget or time.time() > deadline:
                break

        return best_x, best_f
