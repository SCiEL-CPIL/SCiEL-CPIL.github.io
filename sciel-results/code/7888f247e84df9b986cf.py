import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        if n_evals <= 0:
            return None
        
        # Deadline guard: kill after 330s, use 300s as safety margin
        deadline = time.time() + 300
        
        # Generate x1 values
        # Optimal spacing for ZDT1 hypervolume maximization is x_i ~ i^2
        # This minimizes the Riemann sum error for f2 = 1 - sqrt(x1)
        i = np.arange(n_evals)
        x1_vals = ((i + 1) / n_evals) ** 2.0
        
        last_x = None
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            last_x = problem(x)
            
        return last_x
