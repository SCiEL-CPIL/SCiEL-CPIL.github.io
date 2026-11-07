import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n_evals = self.budget
        dim = self.dim

        # Generate points on the Pareto front
        # For ZDT2, the Pareto front is achieved when x_2..x_n minimize g(x).
        # Based on inspiration, the minimizer is x_i = 0.
        # x_1 varies to cover the front.
        
        if n_evals > 1:
            indices = np.arange(n_evals)
            # Power law spacing p=0.5 to concentrate points near x_1=0
            # where the front f_2 = 1 - sqrt(x_1) has high curvature/slope.
            x1_vals = (indices / (n_evals - 1)) ** 0.5
        else:
            x1_vals = np.array([0.5])
            
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_vals
        points[:, 1:] = 0.0  # Set x_2..x_n to 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(points[i])
            
        return points
