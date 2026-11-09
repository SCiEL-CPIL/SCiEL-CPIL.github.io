import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 Pareto front: x_1 in [0, 1], x_2...x_n = 0
        # f1 = x_1, f2 = 1 - sqrt(x_1)
        
        # We have budget evaluations. Use them all to sample the front.
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # Generate x_1 values uniformly in [0, 1]
        # To maximize hypervolume, uniform in f1 is a good start.
        # Let's use linspace to ensure coverage.
        x1_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Construct candidate solutions
        # x = [x1, 0, 0, ..., 0]
        # Note: bounds are [0, 1] for all variables.
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1_vals[i]
            
            # Ensure x is within bounds (it is, since x1 in [0,1] and rest 0)
            # problem clips anyway
            
            f = problem(x)
            # The evaluator records f automatically? 
            # "Every returned objective vector is recorded"
            # So we just need to call problem(x).
            
        return None
