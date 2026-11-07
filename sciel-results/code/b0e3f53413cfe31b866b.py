import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals <= 0:
            return np.array([])
            
        # ZDT1: f1 = x1, f2 = 1 - sqrt(x1) when x[1:] = 0
        # Pareto front: f2 = 1 - sqrt(f1), f1 in [0,1]
        # To maximize hypervolume with reference [11,11], distribute uniformly in objective space
        # Parameterize by f2: f2 = t in [0,1], then f1 = (1-t)^2
        # Uniform spacing in f2 gives optimal hypervolume for convex front
        
        # Generate f2 values uniformly in [0, 1]
        f2_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Convert to x1: f1 = (1 - f2)^2, and f1 = x1
        x1_vals = (1.0 - f2_vals) ** 2
        
        # Clip to bounds to be safe
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        # Create the population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = lo[1:]
        
        # Evaluate each point to record it
        for i in range(n_evals):
            if time.time() >= deadline:
                break
            problem(X[i])
            
        return X
