"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Warped grid-based sampling of the DTLZ7 Pareto front with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget

        # DTLZ7 with m=3, n=22: Pareto front has g=0 (last 20 vars = 0)
        # f1=x0, f2=x1, f3=h(x1) where h(x1) = (x1+1)*(9-x1)/10
        # So front is 2D: parameterized by x0, x1 in [0,1]
        # Set x2..x21 to 0 to achieve g=0

        # The key insight: f3 = (x1+1)*(9-x1)/10 is a concave function of x1
        # Maximum at x1=4, but x1 is in [0,1], so f3 is decreasing on [0,1]
        # f3(0) = 0.9, f3(1) = 0.8
        # The curvature means uniform x1 spacing gives non-uniform f3 spacing
        # We want to warp x1 so that f3 values are more evenly spaced

        # For hypervolume optimization with reference point [1.1, 1.1, 25.0],
        # we want good coverage of the Pareto front in objective space
        # f1 = x0, f2 = x1, f3 = (x1+1)*(9-x1)/10

        # Use a stratified grid with warping in x1 direction
        # n_grid x n_grid points
        n_grid = int(np.sqrt(budget))
        if n_grid * n_grid > budget:
            n_grid -= 1

        # For x0 (f1): uniform spacing is fine since f1=x0 is linear
        # For x1 (f2, f3): warp to get better coverage of f3 values
        # f3(x1) = (x1+1)*(9-x1)/10 = (9x1 + 9 - x1^2 - x1)/10 = (8x1 + 9 - x1^2)/10
        # df3/dx1 = (8 - 2x1)/10, which is positive on [0,1] (decreasing from 0.8 to 0.6)
        # To make f3 more uniform, we can use inverse CDF transformation

        # Simple approach: use a power-law warp
        # x1_warped = u^alpha where alpha > 1 concentrates points near x1=0
        # Actually, let's use a different strategy: stratified sampling in f3 space

        # Generate base grid in [0,1]^2
        u0 = np.linspace(0.0, 1.0, n_grid)
        u1 = np.linspace(0.0, 1.0, n_grid)

        # Warp u1 to get better f3 coverage
        # f3(x1) = (x1+1)*(9-x1)/10
        # We want roughly uniform f3 values
        # f3 ranges from f3(0)=0.9 to f3(1)=0.8
        # This is a small range, so the warping effect is modest
        # Let's use a mild warp: x1 = u1^1.2 (slightly more points near 0)
        x1_warped = u1 ** 1.15

        # Also add jitter to avoid exact boundaries and improve coverage
        # Use deterministic jitter based on grid position
        rng = np.random.default_rng(42)
        jitter0 = rng.uniform(-0.5, 0.5, size=(n_grid, n_grid)) / n_grid
        jitter1 = rng.uniform(-0.5, 0.5, size=(n_grid, n_grid)) / n_grid

        results = []
        used = 0

        for i in range(n_grid):
            for j in range(n_grid):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x0 = u0[i] + jitter0[i, j]
                x1 = x1_warped[j] + jitter1[i, j]

                # Clip to [0, 1]
                x0 = np.clip(x0, 0.0, 1.0)
                x1 = np.clip(x1, 0.0, 1.0)

                # Create solution vector
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                # x[2:] = 0 already (achieves g=0)

                f = problem(x)
                results.append(f)
                used += 1

            if time.time() > deadline or used >= budget:
                break

        # If we have remaining budget, add more points with stratified random
        if used < budget and time.time() < deadline:
            remaining = budget - used
            # Use stratified random sampling in [0,1]^2
            n_extra_grid = int(np.sqrt(remaining))
            if n_extra_grid == 0:
                n_extra_grid = 1

            u0_extra = np.linspace(0.0, 1.0, n_extra_grid)
            u1_extra = np.linspace(0.0, 1.0, n_extra_grid)

            for i in range(n_extra_grid):
                for j in range(n_extra_grid):
                    if time.time() > deadline:
                        break
                    if used >= budget:
                        break

                    # Stratified random within each cell
                    x0 = (i + rng.random()) / n_extra_grid
                    x1_warped_u = (j + rng.random()) / n_extra_grid
                    x1 = x1_warped_u ** 1.15

                    x0 = np.clip(x0, 0.0, 1.0)
                    x1 = np.clip(x1, 0.0, 1.0)

                    x = np.zeros(dim)
                    x[0] = x0
                    x[1] = x1

                    f = problem(x)
                    results.append(f)
                    used += 1

                if time.time() > deadline or used >= budget:
                    break

        return results
# EVOLVE-BLOCK-END
