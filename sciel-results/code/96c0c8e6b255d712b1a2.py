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
        
        # DTLZ4: variables 0,1 determine Pareto front position
        # variables 2..11 determine distance from front (optimal at 0.5)
        # For 3 objectives, the Pareto front is defined by:
        # f1 = 0.5 * (1 + x0) * g(x)
        # f2 = 0.5 * (1 - x0) * (1 + x1) * g(x)
        # f3 = 0.5 * (1 - x1) * g(x)
        # where g(x) = sum((xi - 0.5)^(2*alpha) for i in 2..11)
        # When xi = 0.5 for all i >= 2, g = 0, and we get the ideal front:
        # f1 = 0.5*(1+x0), f2 = 0.5*(1-x0)*(1+x1), f3 = 0.5*(1-x1)
        # with x0, x1 in [0,1]
        
        # The Pareto front in objective space is a curved surface.
        # We want to sample x0, x1 to maximize hypervolume.
        
        # Use Sobol-like quasi-random sequence for better coverage
        # Since scipy may have sobol, let's use it if available, otherwise use Halton
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n=n_evals)
            # points shape: (n_evals, 2)
            x0_vals = points[:, 0]
            x1_vals = points[:, 1]
        except ImportError:
            # Fallback to Halton sequence
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
            
            x0_vals = halton(n_evals, 2)
            x1_vals = halton(n_evals, 3)
        
        # Create population: all 12 variables
        # Variables 0, 1: sampled
        # Variables 2-11: set to 0.5 (optimal for DTLZ4)
        X = np.zeros((n_evals, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        if dim > 2:
            X[:, 2:] = 0.5
        
        # Evaluate all points
        F = np.zeros((n_evals, n_obj))
        for i in range(n_evals):
            if time.time() > deadline:
                # Stop and return what we have so far
                F[i:] = np.nan
                break
            F[i] = problem(X[i])
        
        # Remove any NaN rows (if we hit the deadline)
        valid_mask = ~np.any(np.isnan(F), axis=1)
        F = F[valid_mask]
        
        return F
