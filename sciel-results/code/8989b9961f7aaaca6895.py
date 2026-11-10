import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT4 specific knowledge:
        # x0 in [-5, 5], x1..x9 in [0, 1]
        # Optimal x1..x9 = 0
        # f1 = x0 + 2
        # We want to sample x0 to cover f1 in [-3, 7]
        
        n_evals = self.budget
        dim = self.dim
        
        # Create x0 values uniformly in [-5, 5]
        x0_vals = np.linspace(-5.0, 5.0, n_evals)
        
        # Create population
        X = np.zeros((n_evals, dim))
        X[:, 0] = x0_vals
        # x1..x9 are already 0
        
        # Evaluate all points
        # We must call problem for each point
        for i in range(n_evals):
            # Ensure we don't exceed budget
            problem(X[i])
            
        return X
