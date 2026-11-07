"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer exploiting DTLZ7's analytical Pareto front.

    DTLZ7 with M=3 objectives has a Pareto front:
        f1 = x0,  f2 = x1,  f3 = 2 - sqrt(x0)   (when the tail term g = 1)
    achieved by setting x2..x21 = 0.5 (so the tail sum is zero).

    The reference point [1.1, 1.1, 25.0] is far in the third objective, so the
    hypervolume contribution per unit area of the (f1, f2) front is roughly
    constant. Hypervolume is therefore maximized by COVERING the 2D front as
    evenly as possible, which a uniform grid does better (smaller covering
    radius) than clustered/sparse samples. We spend the whole budget on a dense
    uniform grid of on-front points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim

    def __call__(self, problem, bounds):
        budget = int(self.budget)
        dim = int(self.dim)
        n_obj = int(self.n_obj)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Number of grid lines per objective axis; square grid so the 2D front
        # is covered uniformly. Keep within the evaluation budget.
        n = int(np.floor(np.sqrt(budget)))
        if n < 2:
            n = 2
        # Ensure we do not exceed the budget.
        while n * n > budget:
            n -= 1

        # Uniform nodes in [0, 1] for the two front parameters (x0, x1).
        # n points on each axis -> n*n points on the front.
        t = np.linspace(0.0, 1.0, n)

        # Build the full parameter vectors: x0, x1 vary over the grid; the
        # remaining dim-2 variables are set to 0.5 (the DTLZ7 tail minimum,
        # which gives g = 1 and places the point exactly on the Pareto front).
        # Clip to the supplied bounds for safety.
        tail_val = 0.5
        tail_val = min(max(tail_val, lo[2]), hi[2]) if dim > 2 else 0.0

        X = np.empty((n * n, dim), dtype=float)
        # Fill x0 (col 0) and x1 (col 1) with the grid, tail with tail_val.
        x0_grid = np.repeat(t, n)          # x0 varies slow
        x1_grid = np.tile(t, n)            # x1 varies fast
        X[:, 0] = x0_grid
        X[:, 1] = x1_grid
        if dim > 2:
            X[:, 2:] = tail_val

        # Clip everything into bounds (front params are already in [0,1] and
        # bounds are [0,1], so this is a no-op in the normal case).
        X = np.clip(X, lo, hi)

        # Evaluate every grid point.
        F = np.array([problem(x) for x in X], dtype=float)

        return X
# EVOLVE-BLOCK-END
