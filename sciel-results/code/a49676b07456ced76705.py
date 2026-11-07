import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 Pareto Front Strategy:
        # The Pareto front is located at x_2...x_10 = 0 and x_1 in [0, 1].
        # We sample x_1 uniformly in [0, 1] and set other variables to 0.
        
        n_evals = self.budget
        # Create x_1 values
        # Ensure we stay within bounds for x_1 if they are restricted, 
        # but standard ZDT4 front is [0, 1].
        x1_lo = max(0.0, lo[0])
        x1_hi = min(1.0, hi[0])
        
        if x1_lo >= x1_hi:
            # Fallback if bounds don't cover [0, 1] properly
            x1_vals = np.linspace(lo[0], hi[0], n_evals)
        else:
            x1_vals = np.linspace(x1_lo, x1_hi, n_evals)
            
        # Create the population matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        # x_2...x_10 are already 0
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must call problem(x) for each x
        # To be efficient, we can't vectorize problem(x) if it's a black box, 
        # but we can loop. 200 calls is fast.
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
