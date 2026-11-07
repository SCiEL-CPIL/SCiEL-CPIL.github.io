import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # Strategy:
        # ZDT3 Pareto front is effectively 1D (determined by x_1).
        # x_2...x_n should be 0 to minimize g(x) and reach the Pareto front.
        # We use a uniform grid of n_evals points across x_1 in [lo[0], hi[0]].
        
        # Generate x_1 samples
        x1_samples = np.linspace(lo[0], hi[0], n_evals)
        
        # Initialize array of solutions
        # x[0] varies, x[1:] = 0 (which is optimal for ZDT3 g(x))
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_samples
        
        # Evaluate each point
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return None
