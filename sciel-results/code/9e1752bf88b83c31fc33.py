"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = 300  # Safety margin before 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # We have 500 evaluations total.
        # Let's use 5 multi-starts.
        # Each start gets 100 evaluations for L-BFGS-B.
        n_starts = 5
        evals_per_start = self.budget // n_starts
        
        # Generate start points
        # One at center, others random in [-4, 4]
        start_points = []
        start_points.append(np.zeros(self.dim))
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, self.dim))
            
        for x0 in start_points:
            # Check deadline
            if time.time() > self.start_time + self.deadline:
                break
                
            # Define a wrapper to count evaluations and check deadline
            eval_count = 0
            
            def wrapped_func(x):
                nonlocal eval_count
                if time.time() > self.start_time + self.deadline:
                    raise TimeoutError("Deadline exceeded")
                eval_count += 1
                return func(x)
                
            try:
                # L-BFGS-B is good for this.
                # Bounds are [-5, 5]
                bounds = [(-5, 5)] * self.dim
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=bounds, 
                               options={'maxiter': evals_per_start, 'maxfun': evals_per_start, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception as e:
                # If an exception occurs, continue to next start
                continue
                
        # If no valid result, return a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
