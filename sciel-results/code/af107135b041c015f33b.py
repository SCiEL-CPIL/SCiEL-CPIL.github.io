"""Global grid exploration followed by targeted local search for BBOB f20."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based global exploration + local refinement for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid exploration to identify promising regions
        # Use a stratified grid to cover the domain systematically
        # For 5D, use 3^5 = 243 points for coarse grid
        n_per_dim = 3
        grid_calls = n_per_dim ** dim
        if grid_calls > budget // 3:
            # Reduce grid size if budget is tight
            n_per_dim = 2
            grid_calls = n_per_dim ** dim

        best_x = None
        best_f = np.inf
        call_count = 0

        # Generate grid points
        grid_points = []
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    for l in range(n_per_dim):
                        for m in range(n_per_dim):
                            x = np.array([
                                -5 + 10 * i / (n_per_dim - 1) if n_per_dim > 1 else 0,
                                -5 + 10 * j / (n_per_dim - 1) if n_per_dim > 1 else 0,
                                -5 + 10 * k / (n_per_dim - 1) if n_per_dim > 1 else 0,
                                -5 + 10 * l / (n_per_dim - 1) if n_per_dim > 1 else 0,
                                -5 + 10 * m / (n_per_dim - 1) if n_per_dim > 1 else 0,
                            ])
                            grid_points.append(x)

        # Evaluate grid points
        for x in grid_points:
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from best grid point using adaptive step
        if best_x is not None and call_count < budget:
            # Use a simple adaptive local search
            sigma = 1.0
            no_improve = 0
            max_no_improve = 50

            while call_count < budget:
                if time.time() > deadline:
                    break
                if no_improve > max_no_improve:
                    break

                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    no_improve += 1
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
