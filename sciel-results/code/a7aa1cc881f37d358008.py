import numpy as np
import time

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

        # Generate LHS points
        # For each dimension, divide [0, 1] into N equal intervals.
        # Pick one random point in each interval.
        # Then permute the rows for each dimension to decorrelate them.
        
        # Optimization: Vectorize LHS generation
        # intervals: (N, 1)
        # random offsets: (N, 1)
        # base: (N, dim)
        
        # Standard LHS:
        # 1. For each dim d, create a vector of N values: (i + U_i) / N
        # 2. Permute each dim independently.
        
        # To be efficient:
        # Create an array of shape (dim, N)
        # For each row d:
        #   vals = (np.arange(N) + np.random.rand(N)) / N
        #   np.random.shuffle(vals)
        
        # But we need to be careful with the random state.
        # The evaluator seeds np.random.
        
        X_lhs = np.zeros((N, self.dim))
        for d in range(self.dim):
            # (i + U_i) / N
            # np.arange(N) is 0..N-1
            # np.random.rand(N) is U_i
            # shuffle the result
            vals = (np.arange(N) + np.random.rand(N)) / N
            np.random.shuffle(vals)
            X_lhs[:, d] = vals
        
        # Map to bounds
        X = lo + X_lhs * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = N

        return X
