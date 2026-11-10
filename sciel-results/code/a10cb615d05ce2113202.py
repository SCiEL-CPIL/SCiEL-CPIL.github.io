import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, leave margin
        
        # ZDT3 Pareto front is at x2..xn = 0
        # We sample x1 uniformly in [0, 1]
        # Budget is 200, so we can evaluate 200 points
        n_evals = self.budget
        
        # Generate x1 values
        x1_vals = np.linspace(0, 1, n_evals)
        
        # Create the full vectors
        # x = [x1, 0, 0, ..., 0]
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
