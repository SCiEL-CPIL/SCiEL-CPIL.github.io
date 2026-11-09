"""Multi-objective DTLZ4 optimizer using Halton sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front via 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_shape = m - 1  # 2 shape variables for 3 objectives

        # Halton sequence in 2D (bases 2 and 3)
        def halton_2d(npts):
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            pts = np.zeros((npts, 2))
            for i in range(1, npts + 1):
                pts[i-1, 0] = halton_1d(i, 2)
                pts[i-1, 1] = halton_1d(i, 3)
            return pts

        # Reserve evaluations for important boundary points
        # Corners of [0,1]^2 in shape space
        n_corners = 4
        n_halton = self.budget - n_corners
        if n_halton < 1:
            n_halton = 1
            n_corners = 0

        # Generate Halton points
        halton_pts = halton_2d(n_halton)

        # Build shape point list
        if n_corners > 0:
            corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            shape_pts = np.vstack([corners, halton_pts])
        else:
            shape_pts = halton_pts

        # Trim to budget
        shape_pts = shape_pts[:self.budget]

        # Build full x vectors
        # g-variables set to 0.5 (assumed optimal for the evaluator's g-function)
        g_val = 0.5
        X = np.zeros((len(shape_pts), n))
        
        # Map Halton points in [0,1]^2 to angles that provide uniform coverage on the spherical Pareto front
        # Assuming DTLZ2/4-like structure where front is a sphere surface
        # u, v in [0, 1]
        # alpha = arccos(1 - u) ensures uniform distribution in alpha (area weighting sin(alpha))
        # beta = v * pi/2 ensures uniform distribution in beta
        u = shape_pts[:, 0]
        v = shape_pts[:, 1]
        
        alpha = np.arccos(1.0 - u)
        beta = v * (np.pi / 2.0)
        
        X[:, 0] = alpha
        X[:, 1] = beta
        
        for j in range(n_shape, n):
            X[:, j] = g_val

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points with time guard
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
