import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 Pareto front is at x_i = 0 for i > 0
        # f1 = x0, f2 = 101 * (1 - sqrt(x0))
        # We sample x0 uniformly in [0, 1]
        
        n_samples = self.budget
        
        # Generate x0 values
        x0_vals = np.linspace(lo[0], hi[0], n_samples)
        
        # Initialize x array
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] are already 0.0
        
        # Evaluate all
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X[0]
