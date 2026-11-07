import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # 1. Latin Hypercube Sampling (LHS) for initial coverage
        # This ensures that every dimension is uniformly covered in the first pass.
        # For 200 evaluations in 12 dimensions, a single pass of LHS provides
        # much better coverage of the search space than random sampling,
        # which is critical for capturing the disconnected Pareto front of WFG9.
        
        N = self.budget
        if N == 0:
            return np.array([])

        # Use Sobol sequence for better coverage than random LHS
        # Sobol sequences are quasi-random and have better low-discrepancy properties,
        # which is crucial for capturing the disconnected Pareto front of WFG9
        # with a limited budget.
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            X_lhs = sampler.random(N)
        except Exception:
            # Fallback to LHS if Sobol fails
            X_lhs = np.zeros((N, self.dim))
            for d in range(self.dim):
                vals = (np.arange(N) + np.random.rand(N)) / N
                np.random.shuffle(vals)
                X_lhs[:, d] = vals
        
        # Map to bounds
        X = lo + X_lhs * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = N

        return X
