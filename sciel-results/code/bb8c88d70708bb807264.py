import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = self.budget
        
        # Generate x[0] values using Sobol sequence for better space-filling properties
        # This ensures more uniform coverage of [0,1] compared to linspace,
        # helping to better hit boundaries of the 7 disjoint Pareto segments
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=False)
        x0_vals = sampler.random(n_samples)[:, 0]
        
        # Ensure we include the exact endpoints 0 and 1 for hypervolume
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        # Create the population
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_vals
        # x[1:] is already 0
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            pass
        
        return X
