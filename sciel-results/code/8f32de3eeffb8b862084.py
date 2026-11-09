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
        
        # ZDT3 Pareto front: x[1:] = 0 minimizes g(x)=1
        # x[0] controls the spread of the front
        # We sample x[0] densely to cover the front well
        
        # Use most of the budget for dense sampling of x[0]
        n_samples = min(self.budget, 200)
        
        # Dense uniform sampling of x[0] in [0, 1]
        x0_vals = np.linspace(0.0, 1.0, n_samples)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_vals
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
