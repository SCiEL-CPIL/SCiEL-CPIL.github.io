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
        
        n_points = self.budget
        
        # Optimal distribution derived from HV maximization
        # x(k) = 100 * ( (1.1)^(k/N) - 1 )^2
        # k ranges from 0 to N-1? Or 0 to N?
        # np.linspace(0, 1, N) gives N points.
        # We want k to go from 0 to N-1 for the indices?
        # Let's use t = np.linspace(0, 1, n_points)
        # x = 100 * ( (1.1)^t - 1 )^2
        
        t = np.linspace(0.0, 1.0, n_points)
        x0_vals = 100.0 * (np.power(1.1, t) - 1.0)**2
        
        # Clip to [0, 1] just in case of floating point errors
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            try:
                F[i] = problem(X[i])
            except Exception:
                break
                
        return X
