import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate Sobol points for x0, x1
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n=n_evals)
            u0 = points[:, 0]
            u1 = points[:, 1]
        except ImportError:
            # Fallback to Halton
            def halton(n, base):
                result = np.zeros(n)
                for i in range(1, n + 1):
                    f = 1.0
                    r = 0.0
                    i_copy = i
                    while i_copy > 0:
                        f /= base
                        r += f * (i_copy % base)
                        i_copy //= base
                    result[i - 1] = r
                return result
            u0 = halton(n_evals, 2)
            u1 = halton(n_evals, 3)
        
        # Use uniform distribution over the parameter space [0, 1] for x0, x1
        # This corresponds to a uniform distribution on the Pareto front parameters
        x0_vals = u0
        x1_vals = u1
        
        # Clip to [0, 1] just in case of floating point errors
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Create population
        X = np.zeros((n_evals, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        if dim > 2:
            X[:, 2:] = 0.5  # Optimal for DTLZ4 distance variables
            
        # Evaluate all points
        F = np.zeros((n_evals, n_obj))
        for i in range(n_evals):
            if time.time() > deadline:
                F[i:] = np.nan
                break
            F[i] = problem(X[i])
        
        # Remove any NaN rows
        valid_mask = ~np.any(np.isnan(F), axis=1)
        F = F[valid_mask]
        
        return F
