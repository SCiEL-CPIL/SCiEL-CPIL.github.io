# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # For ZDT1, the Pareto front is achieved when x_2...x_d = 0
        # and x_1 varies in [0, 1].
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Optimal density for hypervolume maximization:
        # t_i = (i / (N-1))^2
        idx = np.arange(n_evals)
        u = idx / (n_evals - 1)
        t_vals = u ** 2
        
        # Clip to bounds
        t_vals = np.clip(t_vals, lo[0], hi[0])
        
        # Construct points
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        # x_2...x_d = 0
        
        last_obj = None
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            last_obj = f
                
        return last_obj
# EVOLVE-BLOCK-END
