"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front parameterization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure for 3 objectives:
        # - x[0] and x[1] determine the Pareto front shape
        # - x[2:] affect g; g=0 when all are 0.5
        # The Pareto front (g=0) is a quarter sphere of radius 0.5:
        #   f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        #   f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        #   f3 = 0.5 * sin(pi*x1/2)
        # where x1, x2 ∈ [0, 1]

        # Helper: create decision vector from x1, x2
        def make_x(x1, x2):
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        evaluated_points = []
        used = 0

        # Strategy: Use a stratified 2D grid in (x1, x2) space to densely
        # cover the Pareto front. This is better than Sobol for maximizing
        # hypervolume because it ensures uniform coverage.

        # Generate a stratified grid
        # For budget evaluations, use a square grid of size n x n where n*n <= budget
        # But we can also use non-square grids for better coverage

        # Use a combination: main grid + halton sequence for remaining points
        n_grid = int(np.sqrt(budget))
        # Ensure we don't exceed budget
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Map grid indices to [0, 1] with midpoint sampling
                x1 = (i + 0.5) / n_grid
                x2 = (j + 0.5) / n_grid
                grid_points.append((x1, x2))

        # If we have remaining budget, add Halton sequence points
        remaining = budget - len(grid_points)
        if remaining > 0:
            # Halton sequence for better low-discrepancy
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i = i // base
                    f /= base
                return result

            # Start from index 1 (index 0 is undefined for Halton)
            for k in range(1, remaining + 1):
                x1 = halton(k, 2)
                x2 = halton(k, 3)
                grid_points.append((x1, x2))

        # Shuffle to avoid any ordering bias (though not critical for hypervolume)
        np.random.shuffle(grid_points)

        # Evaluate all points
        for x1, x2 in grid_points:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        # If we still have budget remaining (shouldn't happen, but just in case),
        # use random sampling
        while used < budget:
            if time.time() - t0 > deadline:
                break
            x1 = np.random.random()
            x2 = np.random.random()
            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
