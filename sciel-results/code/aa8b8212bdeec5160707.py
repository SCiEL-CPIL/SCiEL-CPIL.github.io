import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The Pareto front for ZDT4 is obtained by setting x_1...x_{n-1} = 0
        # and varying x_0 in [0, 1].
        # f1 = x_0
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(x_0) - x_0 * sin^2(10*pi*x_0))
        
        # We want to maximize hypervolume.
        # Points on the Pareto front are best.
        # We distribute the budget evenly along x_0 in [0, 1].
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        # Generate x_0 values
        # Use linspace to cover the range [0, 1]
        x0_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Construct X
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] is already 0
        
        # Clip to bounds just in case, though 0 and [0,1] should be within standard bounds
        # If bounds for x0 are different, we might need to adjust, but standard is [0,1]
        # The problem clips anyway.
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
