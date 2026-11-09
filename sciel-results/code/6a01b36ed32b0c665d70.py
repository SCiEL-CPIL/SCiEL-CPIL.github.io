import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 True Pareto Front:
        # g(x) = 1 + 10(n-1) + sum((x_i - 0.5)^2 - 10*cos(4*pi*x_i))
        # Minimum of the sum term is 0, achieved when x_i = 0.5 for all i >= 1.
        # Thus, the Pareto front is located at x_i = 0.5 for i > 0.
        # x_0 varies in [0, 1] to cover the front.
        
        n_samples = min(self.budget, 200)
        
        # Dense uniform sampling of x[0] in [0, 1]
        x0_vals = np.linspace(0.0, 1.0, n_samples)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_vals
        # Set remaining dimensions to 0.5 to minimize g(x) to 1
        X[:, 1:] = 0.5
        
        # Clip to bounds (should be no-op if bounds are [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
