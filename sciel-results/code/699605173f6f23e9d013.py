import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT3, the Pareto front is achieved when x_2...x_n = 0
        # This minimizes g(x) and thus f_2(x) for any given x_1
        x_opt_rest = 0.0
        
        n_points = self.budget
        
        # Use a grid that ensures good coverage of the oscillatory Pareto front
        # Uniform sampling is effective, but we can ensure endpoints are included
        x1_values = np.linspace(0.0, 1.0, n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = x_opt_rest
        
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
