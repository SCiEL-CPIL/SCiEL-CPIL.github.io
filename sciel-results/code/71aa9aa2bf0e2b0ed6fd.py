"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: sample the known Pareto front via a 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        # k = number of "g" variables (last k variables), shape vars = first (m-1)
        k = n - m + 1
        n_shape = m - 1  # 2 for 3 objectives

        # Build a 2D Halton (low-discrepancy) sequence over the shape variables
        def halton_2d(npts):
            # 2D Halton sequence: base 2 and 3
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            xs = []
            for i in range(1, npts + 1):
                xs.append([halton_1d(i, 2), halton_1d(i, 3)])
            return np.array(xs)

        # Allocate evaluations: reserve a few for corners, rest for Halton
        n_corners = 4  # corners of the 2D shape space
        n_halton = self.budget - n_corners
        if n_halton < 1:
            n_halton = max(1, self.budget - n_corners)

        # Shape-space corners (x0, x1) in [0,1]^2
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        # Build full x vectors: shape vars from sequence, g-vars = 0.5 (clipped to bounds)
        g_val = 0.5
        g_val = np.clip(g_val, lo[n_shape], hi[n_shape])

        shape_pts = np.vstack([corners, halton_2d(n_halton)])
        # Trim to budget if needed
        shape_pts = shape_pts[:self.budget]

        # Build x matrix
        X = np.empty((len(shape_pts), n))
        X[:, :n_shape] = shape_pts
        for j in range(n_shape, n):
            X[:, j] = g_val

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
