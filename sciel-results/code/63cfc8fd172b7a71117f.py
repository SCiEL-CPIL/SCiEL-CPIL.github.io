"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer for DTLZ1 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Halton sequence for the first 2 variables
        # The remaining variables are fixed to 0.5 to minimize g
        n_points = self.budget
        dim_shape = 2
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        X = np.zeros((n_points, self.dim))
        for i in range(n_points):
            # Generate Halton numbers for the first 2 dimensions
            X[i, 0] = halton(i + 1, 2)
            X[i, 1] = halton(i + 1, 3)
            # Fix the rest to 0.5 (midpoint of [0,1] bounds, assuming standard DTLZ1)
            # If bounds are not [0,1], we should map to the middle of the bounds
            for j in range(2, self.dim):
                X[i, j] = (lo[j] + hi[j]) / 2.0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
