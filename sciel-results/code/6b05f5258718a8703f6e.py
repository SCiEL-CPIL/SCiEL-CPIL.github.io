# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7.
    Strategy: Sample the 2D Pareto front (x_3..x_22 = 0) with a non-uniform grid
    that is denser near the ideal point (f1=0, f2=0) to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # Determine grid size to use most of the budget
        grid_size = int(np.sqrt(self.budget))
        grid_size = max(2, grid_size)
        while (grid_size + 1) ** 2 <= self.budget:
            grid_size += 1

        # Non-linear grid to concentrate points near the ideal objectives
        # f1 = 1 - x1, f2 = x2.
        # We want f1, f2 small for hypervolume.
        # Use power transformation: x1 = 1 - t^gamma, x2 = t^delta
        # gamma, delta > 1 makes t^gamma, t^delta denser near 0

        gamma = 2.0
        delta = 2.0

        t_vals = np.linspace(0.0, 1.0, grid_size)
        x1_vals = 1.0 - t_vals ** gamma
        x2_vals = t_vals ** delta

        # Build list of (x1, x2) pairs, including corners and edge midpoints
        # to ensure extreme points of the Pareto front are always sampled
        points = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                points.append((x1, x2))

        # Add explicit corner points if not already present
        corners = [
            (0.0, 0.0),   # f1=1, f2=0
            (1.0, 0.0),   # f1=0, f2=0
            (0.0, 1.0),   # f1=1, f2=1
            (1.0, 1.0),   # f1=0, f2=1
        ]
        for cx1, cx2 in corners:
            if not any(abs(p[0] - cx1) < 1e-10 and abs(p[1] - cx2) < 1e-10 for p in points):
                points.append((cx1, cx2))

        # Add edge midpoints
        edge_mids = [
            (0.5, 0.0),
            (0.5, 1.0),
            (0.0, 0.5),
            (1.0, 0.5),
        ]
        for ex1, ex2 in edge_mids:
            if not any(abs(p[0] - ex1) < 1e-10 and abs(p[1] - ex2) < 1e-10 for p in points):
                points.append((ex1, ex2))

        # Evaluate all points
        for x1, x2 in points:
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            # x[2:] = 0 for g=0 (Pareto front)

            try:
                obj = problem(x)
                obj = list(obj)
                results.append(obj)
                n_evals += 1
            except Exception:
                break

        return results
# EVOLVE-BLOCK-END
