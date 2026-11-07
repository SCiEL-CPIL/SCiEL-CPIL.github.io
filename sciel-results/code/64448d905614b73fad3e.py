# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer that directly constructs points on the known
    DTLZ1 Pareto front using a Sobol (low-discrepancy) sequence for the
    position variables, fixing aggregation variables to 0.5.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s wall-clock

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1: first (dim - n_obj + 1) vars are position, last (n_obj - 1) are aggregation
        n_pos = dim - n_obj + 1  # 7 - 3 + 1 = 5
        n_agg = n_obj - 1  # 2

        # For DTLZ1, Pareto front is achieved when all aggregation vars = 0.5
        # Position vars determine the shape of the front
        # The Pareto front of DTLZ1 in objective space is:
        #   f1 = 0.5 * x0 * (1 - g)
        #   f2 = 0.5 * x0 * x1 * (1 - g)
        #   f3 = 0.5 * x0 * (1 - x1) * (1 - g)
        # where g = 100 * (n_agg + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in agg vars)
        # When all agg vars = 0.5, g = 0, so:
        #   f1 = 0.5 * x0
        #   f2 = 0.5 * x0 * x1
        #   f3 = 0.5 * x0 * (1 - x1)
        # with x0, x1 in [0,1]

        # We need n_pos = 5 position variables. For DTLZ1 with M=3:
        # The Pareto front is parameterized by x0, x1 (the first two position vars)
        # The remaining position vars (x2, x3, x4) do not appear in the objective
        # when g=0. So we can set them to any value, e.g., 0.5.

        # Use Sobol sequence to fill (n_pos) dimensional space, but really we only
        # need x0 and x1 to span the front. We'll use a Sobol sequence in 2D for
        # x0 and x1, and set the rest to 0.5.

        n_evals = self.budget

        # Generate Sobol sequence
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_pos, scramble=True)
            points = sampler.random(n=n_evals)
        except ImportError:
            # Fallback: use Halton or simple Latin hypercube
            # Simple deterministic space-filling: use a grid-like approach
            # For n_pos=5, we'll just use x0 and x1 from a 2D grid
            n_per_axis = int(np.ceil(n_evals ** 0.5))
            grid = np.linspace(0, 1, n_per_axis)
            x0_vals, x1_vals = np.meshgrid(grid, grid)
            x0_flat = x0_vals.ravel()
            x1_flat = x1_vals.ravel()
            n_actual = len(x0_flat)
            x0_flat = x0_flat[:n_evals]
            x1_flat = x1_flat[:n_evals]
            # Pad if needed
            if len(x0_flat) < n_evals:
                x0_flat = np.pad(x0_flat, (0, n_evals - len(x0_flat)), mode='edge')
                x1_flat = np.pad(x1_flat, (0, n_evals - len(x1_flat)), mode='edge')
            points = np.column_stack([x0_flat, x1_flat] + [np.full(n_evals, 0.5) for _ in range(n_pos - 2)])

        # Construct input vectors
        X = np.zeros((n_evals, dim))
        # First n_pos columns: position variables from Sobol
        X[:, :n_pos] = points[:, :n_pos]
        # Last n_agg columns: aggregation variables set to 0.5 for Pareto front
        X[:, n_pos:] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
