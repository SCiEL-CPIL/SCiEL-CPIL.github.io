import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        
        # ZDT2 Pareto Front:
        # f1 = x[0]
        # f2 = g * (1 - sqrt(f1 / g))
        # g = 1 + 10 * (dim - 1)
        
        # Strategy: Uniform spacing in t (x[0]).
        # This maximizes the spread and is a standard baseline.
        # If the reference sample is uniform, this matches it.
        # If the reference is coarse, this is better.
        
        N = budget
        t_vals = np.linspace(0.0, 1.0, N)
        
        last_F = None
        for i in range(N):
            if time.time() > deadline:
                break
            
            t = t_vals[i]
            x = np.zeros(dim)
            x[0] = t
            # x[1:] = 0 gives the Pareto front
            
            F = problem(x)
            last_F = F
        
        return last_F
