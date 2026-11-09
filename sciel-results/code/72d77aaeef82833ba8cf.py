import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Multi-start strategy
        # We will try a few random initial points.
        # L-BFGS-B is very fast for d=5.
        num_restarts = 10
        best_x = None
        best_f = np.inf
        
        # We need to be careful with the budget.
        # scipy's minimize doesn't easily allow us to stop based on a global counter 
        # without wrapping the function.
        
        eval_count = 0
        max_evals_per_run = self.budget // num_restarts + 10 # Add a small buffer
        
        # Wrapper to count evaluations
        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return np.inf
            eval_count += 1
            return func(x)

        for i in range(num_restarts):
            x0 = rng.uniform(-5, 5, self.dim)
            
            # If we've used too much budget, stop
            if eval_count >= self.budget:
                break
            
            # Run optimization
            # maxiter is not the same as maxfev. 
            # For L-BFGS-B, we can set maxiter. 
            # A safe maxiter for d=5 is usually < 100.
            # We can also pass maxfun? No, scipy uses maxiter for L-BFGS-B.
            # Let's just let it run. It will converge quickly.
            
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15})
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        # If no successful run, fall back to best random point? 
        # Or just return the last known best.
        if best_x is None:
             # Fallback: just evaluate a few random points
             for _ in range(min(10, self.budget)):
                 if eval_count >= self.budget: break
                 x = rng.uniform(-5, 5, self.dim)
                 f = func(x)
                 if f < best_f:
                     best_f = f
                     best_x = x
        
        return best_x, best_f
