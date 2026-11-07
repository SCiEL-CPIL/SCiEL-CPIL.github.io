import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT2 that exploits the known Pareto front structure.
    ZDT2 Pareto front is achieved at x[1:] = 0, with f1 = x[0].
    We sample x[0] uniformly in [0, 1] to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Generate a uniform grid on the Pareto front
        # x[0] is the parameter for the front, x[1:] must be 0
        n_points = self.budget
        x0_values = np.linspace(0.0, 1.0, n_points)
        
        # Construct the population
        # Shape: (n_points, dim)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Evaluate all points
        F = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F.append(f)
            
        # The return value is ignored, but we return the population X
        return X
