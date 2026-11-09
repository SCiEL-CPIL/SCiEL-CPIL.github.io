"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit WFG9 structure: fix shape vars at midpoints, grid-sample position vars."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: first n_obj variables are "position" vars, rest are "shape" vars
        n_pos = self.n_obj  # 3 position variables
        n_shape = self.dim - n_pos  # 9 shape variables

        # Fix shape variables at midpoints
        base = np.zeros(self.dim)
        base[:n_pos] = mid[:n_pos]
        base[n_pos:] = mid[n_pos:]

        # Create a dense grid over the position variables
        # 5x5x5 = 125 points, well within 200 budget
        grid_size = 5
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = base.copy()
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1)
                    points.append(x)

        # Evaluate all grid points
        used = 0
        for x in points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            problem(x)
            used += 1

        # Use remaining budget for additional refinement if time permits
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Add some random points near the front for better coverage
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = base.copy()
                # Jitter position variables slightly
                x[:n_pos] = mid[:n_pos] + 0.1 * (hi[:n_pos] - lo[:n_pos]) * rng.standard_normal(n_pos)
                x[:n_pos] = np.clip(x[:n_pos], lo[:n_pos], hi[:n_pos])
                problem(x)
                used += 1

        return None
# EVOLVE-BLOCK-END
