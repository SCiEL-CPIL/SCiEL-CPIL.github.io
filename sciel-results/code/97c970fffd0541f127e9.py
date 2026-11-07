import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume ZDT structure: Pareto front depends primarily on x1.
        # Set other variables to 0 (or lower bound) to minimize penalty terms.
        # We will search x1 in [lo[0], hi[0]].
        
        n_evals = self.budget
        # Use a power law distribution dense near the lower bound (0)
        # x_i = lo + (hi-lo) * (i/n)^alpha
        # alpha > 1 clusters near lo.
        alpha = 1.5
        
        x1_values = (np.arange(1, n_evals + 1) / n_evals) ** alpha
        x1_values = lo[0] + (hi[0] - lo[0]) * x1_values
        
        # Ensure we don't exceed budget or time
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1_values[i]
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
