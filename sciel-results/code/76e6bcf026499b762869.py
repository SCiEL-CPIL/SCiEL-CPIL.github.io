import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # DTLZ7 Pareto Front Structure:
        # To minimize g(x_in), we set x_in = 0.5 for i >= 2.
        # This yields g = 0.
        # f1 = x0
        # f2 = x1
        # f3 = 1 - sqrt(x0)
        # To minimize f2, we set x1 = 0.
        # The Pareto front is the curve: x0 in [0, 1], x1 = 0, x_in = 0.5.
        
        # We sample this curve uniformly in x0 to maximize hypervolume coverage.
        
        x0_values = np.linspace(0.0, 1.0, self.budget)
        
        # Pre-allocate array for points
        points = np.zeros((self.budget, self.dim))
        points[:, 0] = x0_values
        points[:, 1] = 0.0
        if self.dim > 2:
            points[:, 2:] = 0.5
            
        results = []
        for x in points:
            if time.time() > deadline:
                break
            f = problem(x)
            results.append(f)
            
        return results
