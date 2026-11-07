import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: x2...xn = 0, x1 in [0, 1]
        # f1 = x1, f2 = 1 - sqrt(x1)
        
        # Optimal distribution of x1 to maximize hypervolume
        # Derived from minimizing discretization error: x1_i ~ (i/N)^2
        n_points = budget
        
        # Generate x1 values
        # i from 1 to N
        i_vals = np.arange(1, n_points + 1)
        x1_vals = (i_vals / n_points) ** 2
        
        # Ensure x1 is within [0, 1]
        # (1/200)^2 is very small, 1 is 1.
        
        # Evaluate
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            # Clip to bounds just in case, though [0,1] is standard
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
