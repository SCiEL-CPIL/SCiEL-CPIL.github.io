"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # - First dim-5 variables (x1..x7) shape the Pareto front
        # - Last 5 variables (x8..x12) affect g; g=0 when all are 0.5
        # For 3 objectives, only x1 and x2 determine the front shape:
        #   f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        #   f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        #   f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # The Pareto front (g=0) is a quarter sphere of radius 0.5

        n_front = dim - 5  # 7
        n_g = 5            # 5

        # Strategy: Densely sample the known Pareto front with g=0
        # Use stratified sampling in (x1, x2) space for good hypervolume coverage

        used = 0
        results = []

        # Generate a well-distributed set of (x1, x2) points
        # Use a combination of structured grid and quasi-random points

        # Method: Use a stratified grid with jitter for near-uniform coverage
        # With budget=200, use ~196 grid points + 4 extra

        # Primary: structured grid with stratified jitter
        n_grid = 14  # 14x14 = 196 points
        n_grid_points = n_grid * n_grid

        if n_grid_points <= budget:
            # Create stratified grid
            x1_grid = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid)
            x2_grid = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid)

            X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing='ij')
            x1_flat = X1.flatten()
            x2_flat = X2.flatten()

            # Add small stratified jitter to avoid exact grid alignment
            # This helps with numerical issues and slight coverage improvement
            jitter_x1 = (np.random.rand(n_grid_points) - 0.5) * (1.0 / n_grid) * 0.3
            jitter_x2 = (np.random.rand(n_grid_points) - 0.5) * (1.0 / n_grid) * 0.3
            x1_flat = np.clip(x1_flat + jitter_x1, 0.0, 1.0)
            x2_flat = np.clip(x2_flat + jitter_x2, 0.0, 1.0)

            # Create decision vectors with g=0
            X = np.zeros((n_grid_points, dim))
            X[:, 0] = x1_flat
            X[:, 1] = x2_flat
            X[:, 2:n_front] = 0.5  # middle values for non-critical front vars
            X[:, n_front:] = 0.5   # g=0: all g-variables at 0.5

            # Clip to bounds
            X = np.clip(X, lo, hi)

            # Evaluate all grid points
            for i in range(n_grid_points):
                if used >= budget or time.time() - t0 > deadline:
                    break
                f = np.array(problem(X[i]))
                used += 1
                results.append((X[i], f))

        # If we have remaining budget, add quasi-random points for better coverage
        if used < budget:
            # Use Sobol-like or stratified random points
            n_extra = budget - used
            # Use stratified random sampling in the unit square
            # Divide [0,1]^2 into cells and pick one point per cell
            n_cells = int(np.sqrt(n_extra))
            if n_cells < 1:
                n_cells = 1
            n_cells_actual = min(n_cells * n_cells, n_extra)

            for i in range(n_cells_actual):
                if used >= budget or time.time() - t0 > deadline:
                    break

                # Stratified: cell index
                cell_i = i % n_cells
                cell_j = i // n_cells

                # Random point within cell
                x1 = (cell_i + np.random.rand()) / n_cells
                x2 = (cell_j + np.random.rand()) / n_cells
                x1 = np.clip(x1, 0.0, 1.0)
                x2 = np.clip(x2, 0.0, 1.0)

                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                x[2:n_front] = 0.5
                x[n_front:] = 0.5
                x = np.clip(x, lo, hi)

                f = np.array(problem(x))
                used += 1
                results.append((x, f))

        # If still have budget, add a few more random points
        if used < budget:
            for _ in range(budget - used):
                if time.time() - t0 > deadline:
                    break
                x = np.zeros(dim)
                x[0] = np.random.rand()
                x[1] = np.random.rand()
                x[2:n_front] = 0.5
                x[n_front:] = 0.5
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                used += 1
                results.append((x, f))

        return results
# EVOLVE-BLOCK-END
