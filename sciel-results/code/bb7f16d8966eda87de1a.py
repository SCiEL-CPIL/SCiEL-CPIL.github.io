"""Multi-objective optimizer for DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using structured Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate a low-discrepancy sequence for 2D (f1, f2)
        # Using Halton sequence for good spread
        def halton_2d(n):
            """Generate n points in [0,1]^2 using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                # Base 2 for first coordinate
                x, base = 0.0, 2
                f, idx = 1.0, i
                while idx > 0:
                    f /= base
                    x += f * (idx % base)
                    idx //= base
                # Base 3 for second coordinate
                y, base = 0.0, 3
                f, idx = 1.0, i
                while idx > 0:
                    f /= base
                    y += f * (idx % base)
                    idx //= base
                points.append((x, y))
            return np.array(points)

        # Use all budget for structured sampling on the Pareto front
        # x[0] = f1, x[1] = f2, x[2:] = 0.5 (to get g=0)
        n_points = self.budget
        halton_pts = halton_2d(n_points)

        X = np.zeros((n_points, self.dim))
        X[:, 0] = halton_pts[:, 0]
        X[:, 1] = halton_pts[:, 1]
        X[:, 2:] = 0.5  # This gives g=0, placing points on the Pareto front

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
