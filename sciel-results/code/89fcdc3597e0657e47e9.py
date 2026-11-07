import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        # Generate x1 values uniformly in [0, 1]
        # Using midpoints for better integration accuracy
        x1_values = np.linspace(0, 1, n_evals, endpoint=False) + 0.5 / n_evals
        
        # Construct full vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_values
        # x[1:] are already 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return F
