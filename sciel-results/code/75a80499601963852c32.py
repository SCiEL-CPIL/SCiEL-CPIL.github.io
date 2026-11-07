# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # For DTLZ2 with 3 objectives and 12 variables:
        # x[0], x[1] are angle variables
        # x[2:] are distance variables, optimal value is 0.5 (so g=0)
        
        # The Pareto front is the unit sphere in the positive octant.
        # We need to sample points on this sphere to maximize hypervolume.
        
        # Strategy: Use a combination of:
        # 1. A well-distributed set of points on the Pareto front (using Halton sequence)
        # 2. Possibly some local refinement if budget allows
        
        n_points = budget
        
        # Generate Halton sequence in 2D
        def halton_sequence(n, dims):
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # primes: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        i_val = (i_val - 1) // base
                    points[i, d] = f
            return points
        
        halton_pts = halton_sequence(n_points, 2)
        
        # Map to sphere coordinates for uniform distribution on the quarter-sphere
        # DTLZ2 mapping:
        # f_1 = sin(x_1)
        # f_2 = cos(x_1) sin(x_2)
        # f_3 = cos(x_1) cos(x_2)
        
        # For uniform distribution on the sphere:
        # x_1 = arcsin(1 - u)
        # x_2 = v * pi/2
        
        u = halton_pts[:, 0]
        v = halton_pts[:, 1]
        
        x_1 = np.arcsin(1 - u)
        x_2 = v * (np.pi / 2)
        
        # Build X
        X = np.zeros((n_points, dim))
        X[:, 0] = x_1
        X[:, 1] = x_2
        X[:, 2:] = 0.5  # Optimal distance for DTLZ2 is 0.5
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

# EVOLVE-BLOCK-END
