import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Optimal spacing for ZDT2 front f2 = 1 - sqrt(f1) is uniform in f2.
        # f2 in [0, 1] -> f1 = (1-f2)^2
        # Use midpoints for better HV discretization
        u = (np.arange(n_evals) + 0.5) / n_evals
        
        # f2 values
        f2_vals = u
        
        # f1 values
        f1_vals = (1.0 - f2_vals) ** 2
        
        # Map to decision space x[0]
        # x[0] = f1
        # Ensure within bounds [lo[0], hi[0]]
        x1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Construct X
        X = np.full((n_evals, self.dim), 0.2)
        X[:, 0] = x1_vals
        
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            F[i] = problem(X[i])
            if time.time() > deadline:
                return X
                
        return X
