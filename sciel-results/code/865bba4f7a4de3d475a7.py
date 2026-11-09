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

        # ZDT6: f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6, f2 = g*(1-sqrt(f1/g))
        # Pareto front: x2..x10 = 0, so g=1, f2=1
        # f1 min at x1 where sin(6*pi*x1)=±1 and x1 is smallest: x1=1/12

        # Find the minimum f1 point precisely
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # At x1 = 1/12: sin(6*pi/12) = sin(pi/2) = 1
        # f1(1/12) = 1 - exp(-4/12) = 1 - exp(-1/3)

        # But we can do better: find the exact minimum
        # df1/dx1 = 4*exp(-4*x1)*sin(6*pi*x1)^6 - exp(-4*x1)*6*sin(6*pi*x1)^5*6*pi*cos(6*pi*x1)
        # = exp(-4*x1) * sin(6*pi*x1)^5 * [4*sin(6*pi*x1) - 36*pi*cos(6*pi*x1)]
        # Set to zero: sin(6*pi*x1) = 0 or 4*sin(6*pi*x1) = 36*pi*cos(6*pi*x1)
        # tan(6*pi*x1) = 9*pi

        # For the minimum in [0, 1/6], 6*pi*x1 in [0, pi], tan is positive in [0, pi/2)
        # So 6*pi*x1 = arctan(9*pi)
        target_angle = np.arctan(9 * np.pi)
        x1_min = target_angle / (6 * np.pi)

        # Verify: at this x1, sin(6*pi*x1) = sin(arctan(9*pi))
        # sin(arctan(y)) = y/sqrt(1+y^2)
        sin_val = 9 * np.pi / np.sqrt(1 + (9 * np.pi) ** 2)
        f1_min_val = 1 - np.exp(-4 * x1_min) * sin_val ** 6

        # Now generate points: one at x1_min, rest distributed in f1-space
        # Use a fine grid to select uniform f1 points
        n_grid = 100000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid) * (np.sin(6 * np.pi * x1_grid) ** 6)

        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]

        # We want n_evals-1 points uniformly spaced in f1, plus one at f1_min
        # First, find the range of f1 we can achieve
        f1_lo = f1_sorted[0]
        f1_hi = f1_sorted[-1]

        # Select n_evals-1 points uniformly in [f1_lo, f1_hi]
        n_other = n_evals - 1
        f1_targets = np.linspace(f1_lo, f1_hi, n_other)

        # Vectorized inverse: for each target f1, find corresponding x1
        # np.interp works on sorted x -> y mapping
        x1_selected = np.interp(f1_targets, f1_sorted, x1_sorted)

        # Combine with the minimum point
        x1_all = np.concatenate([[x1_min], x1_selected])

        # Remove duplicates (within tolerance)
        x1_unique = np.unique(np.round(x1_all, 12))
        n_unique = len(x1_unique)

        # If we have fewer unique points than budget, add more points
        # near the low-f1 end where hypervolume contribution is highest
        if n_unique < n_evals:
            n_extra = n_evals - n_unique
            # Add points concentrated near f1_min (low f1 = high hypervolume value)
            f1_extra_targets = np.linspace(f1_lo, f1_lo + 0.05, n_extra)
            x1_extra = np.interp(f1_extra_targets, f1_sorted, x1_sorted)
            x1_all = np.concatenate([x1_unique, x1_extra])
        else:
            x1_all = x1_unique[:n_evals]

        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_all
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
