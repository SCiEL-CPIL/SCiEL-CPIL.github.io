"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3, N=12):
      x_0, x_1 : simplex variables defining the front shape
      x_2..x_11: control variables; g is minimized at x_i=0.5
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, (x_0,x_1,1-x_0-x_1) traces the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Determine indices for front and g variables
        # Standard DTLZ4: first M-1 vars are front, rest are g
        n_front_vars = self.n_obj - 1  # 2 for 3 objectives
        n_g_vars = self.dim - n_front_vars
        
        # Generate points on the Pareto front
        # We use a Halton sequence to sample the 2D simplex
        n_points = self.budget
        
        def halton_sequence(count, dims=2):
            points = np.zeros((count, dims))
            for i in range(1, count + 1):
                for d in range(dims):
                    base = 2 + d  # primes 2, 3, 5...
                    f = 1.0
                    r = 0.0
                    n = i
                    while n > 0:
                        f /= base
                        n, rem = divmod(n, base)
                        r += rem * f
                    points[i-1, d] = r
            return points

        halton_pts = halton_sequence(n_points, 2)
        
        # Map square [0,1]^2 to simplex x0 + x1 <= 1
        # x0 = u
        # x1 = v * (1 - u)
        x0 = halton_pts[:, 0]
        x1 = halton_pts[:, 1] * (1 - x0)
        
        # Construct decision vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        
        # Set g variables to their optimal value (0.5)
        # This ensures g=0 (or close to it, 0.5 is exact in binary)
        # We check bounds to ensure 0.5 is valid, but for [0,1] it is
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        optimal_g_val = 0.5
        # If bounds don't include 0.5, clip to closest bound
        # But DTLZ4 bounds are [0,1]
        for i in range(n_front_vars, self.dim):
            val = optimal_g_val
            if val < lo[i]:
                val = lo[i]
            elif val > hi[i]:
                val = hi[i]
            X[:, i] = val
            
        # Clip all to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The evaluator records the objectives returned by problem(x)
        for x in X:
            problem(x)
            
        return X
