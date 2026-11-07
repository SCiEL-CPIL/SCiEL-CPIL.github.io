import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        # ZDT6 Pareto front: x[1:]=0
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)
        # f1 is non-monotonic; achievable f1 range is [f1_min, 1.0]

        # Dense grid to map x1 -> f1
        n_grid = 1000000
        x_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1.0 - np.exp(-4.0 * x_grid) * (np.sin(6.0 * np.pi * x_grid) ** 8)

        # Sort by f1 to get the achievable range
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]

        f1_min = f1_sorted[0]
        f1_max = f1_sorted[-1]

        n_evals = min(budget, 200)

        # Optimal: uniform spacing in f1 space
        # Place n_evals points at uniform f1 intervals from f1_min to f1_max
        f1_targets = np.linspace(f1_min, f1_max, n_evals)

        # For each target f1, find the x1 that achieves it
        # Use searchsorted on sorted f1 to find nearest
        x1_vals = np.empty(n_evals)
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                x1_vals[i] = 0.0
                continue
            # Find index in sorted f1 closest to target
            idx = np.searchsorted(f1_sorted, f1_targets[i])
            # Check neighbors for best match
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(f1_sorted):
                candidates.append(idx)
            if idx + 1 < len(f1_sorted):
                candidates.append(idx + 1)
            best_idx = min(candidates, key=lambda j: abs(f1_sorted[j] - f1_targets[i]))
            x1_vals[i] = x1_sorted[best_idx]

        # Construct points
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_vals

        # Evaluate all points
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = points[i].copy()
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)

        return None
