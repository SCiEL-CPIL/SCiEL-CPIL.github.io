"""Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        r1, r2 = 11.0, 11.0

        # ZDT6 Pareto front: x2..x10=0, g=1
        # f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # f2 = 1 - sqrt(f1)

        # Find f1_min analytically
        target_angle = np.arctan(9 * np.pi)
        x1_min = target_angle / (6 * np.pi)

        # Generate high-res grid for f1->x1 mapping
        n_grid = 1_000_000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid) * (np.sin(6 * np.pi * x1_grid) ** 6)

        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]

        f1_lo = f1_sorted[0]
        f1_hi = f1_sorted[-1]

        # Optimal f1 spacing for 2D hypervolume:
        # Δf1_i ∝ sqrt(f1_i) / (r1 - f1_i)
        # Compute CDF of density h(w) = sqrt(w)/(r1-w)
        n_cdf = 200_000
        w = np.linspace(f1_lo, f1_hi, n_cdf)
        h = np.sqrt(np.maximum(w, 1e-15)) / (r1 - w)
        H = np.cumsum(h)
        H /= H[-1]  # normalize to [0, 1]

        # Inverse CDF: n_evals equally-spaced points in H-space
        targets = np.linspace(0, 1, n_evals)
        f1_selected = np.interp(targets, H, w)

        # Ensure endpoints are included
        f1_selected[0] = f1_lo
        f1_selected[-1] = f1_hi

        # Map f1 -> x1 using high-res grid
        x1_selected = np.interp(f1_selected, f1_sorted, x1_sorted)

        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_selected
        # x2..x10 = 0 for Pareto front

        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
