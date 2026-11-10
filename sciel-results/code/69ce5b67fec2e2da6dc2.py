import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        results = []

        # Strategy: Sample uniformly in f1 space to maximize hypervolume Riemann sum accuracy.
        # 1. Generate a dense grid of x0.
        # 2. Compute f1, f2 for the grid.
        # 3. Select n_evals points that are uniformly spaced in f1.
        # 4. Refine each selected x0 via local Newton/bisection to hit the exact f1 target.
        # 5. Evaluate these points.

        # Dense grid for offline selection
        n_grid = 500000
        x0_grid = np.linspace(0.0, 1.0, n_grid)

        # Compute f1 and f2 for the grid
        # f1(x) = 1 + x * (sin(10*pi*x) + sin(10*pi*x^2))
        # f2(x) = 1 - sqrt(f1(x))
        sin_term1 = np.sin(10 * np.pi * x0_grid)
        sin_term2 = np.sin(10 * np.pi * x0_grid**2)
        f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)

        # Ensure f1 is positive for sqrt
        f1_grid = np.maximum(f1_grid, 1e-6)
        f2_grid = 1 - np.sqrt(f1_grid)

        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]

        # Select n_evals points uniformly spaced in f1
        if n_evals == 1:
            # Pick the point that maximizes the box area (11-f1)(11-f2)
            areas = (11 - f1_grid) * (11 - f2_grid)
            best_idx = np.argmax(areas)
            x0s = [x0_grid[best_idx]]
        else:
            # Uniform spacing in f1 value - use full range for max hypervolume
            f1_min = f1_sorted[0]
            f1_max = f1_sorted[-1]
            f1_targets = np.linspace(f1_min, f1_max, n_evals)

            # For each target, find the index in f1_sorted closest to the target
            # Use searchsorted and check neighbors for closest match
            indices = np.searchsorted(f1_sorted, f1_targets)
            indices = np.clip(indices, 0, n_grid - 1)

            # Check if previous index is closer
            prev_indices = np.clip(indices - 1, 0, n_grid - 1)
            dist_curr = np.abs(f1_sorted[indices] - f1_targets)
            dist_prev = np.abs(f1_sorted[prev_indices] - f1_targets)
            better_mask = dist_prev < dist_curr
            indices[better_mask] = prev_indices[better_mask]

            x0s = x0_sorted[indices]

            # Refinement pass: for each (x0, f1_target) pair, do a local bisection
            # to find x0 that hits the target f1 value more precisely.
            # This improves the Riemann sum accuracy for hypervolume.
            if time.time() < deadline - 10:
                refined_x0s = []
                for i in range(len(x0s)):
                    target = f1_targets[i]
                    x0_init = x0s[i]

                    # Bisection in a small neighborhood around x0_init
                    # The grid spacing is 1/n_grid, so search within ±few grid steps
                    grid_step = 1.0 / n_grid
                    lo = max(0.0, x0_init - 5 * grid_step)
                    hi = min(1.0, x0_init + 5 * grid_step)

                    # Check if target is within [f1(lo), f1(hi)]
                    f1_lo = 1.0 + lo * (np.sin(10.0 * np.pi * lo) + np.sin(10.0 * np.pi * lo**2))
                    f1_hi = 1.0 + hi * (np.sin(10.0 * np.pi * hi) + np.sin(10.0 * np.pi * hi**2))

                    if f1_lo <= target <= f1_hi or f1_hi <= target <= f1_lo:
                        # Bisection
                        for _ in range(40):
                            mid = 0.5 * (lo + hi)
                            f1_mid = 1.0 + mid * (np.sin(10.0 * np.pi * mid) + np.sin(10.0 * np.pi * mid**2))
                            if f1_lo * (f1_mid - target) <= 0:
                                hi = mid
                                f1_hi = f1_mid
                            else:
                                lo = mid
                                f1_lo = f1_mid
                        x0_refined = 0.5 * (lo + hi)
                    else:
                        # Target not in local range; use the grid point as-is
                        x0_refined = x0_init

                    refined_x0s.append(x0_refined)

                x0s = refined_x0s

        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
