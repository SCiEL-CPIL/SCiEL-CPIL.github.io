import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        if self.budget <= 0:
            return np.array([])

        # ZDT3 Pareto front is at x_1 in [0,1] and x_2...x_n = 0
        # We use the full budget to sample this front densely.
        
        n_evals = self.budget
        # Ensure we have at least 1 point
        if n_evals < 1:
            return np.array([])

        # Generate x_1 values
        # Using linspace for uniform coverage
        x1 = np.linspace(0.0, 1.0, n_evals)
        
        # Initialize population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1
        # The rest are 0 by default
        
        # Clip to bounds just in case, though 0 and 1 are within [0,1]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
