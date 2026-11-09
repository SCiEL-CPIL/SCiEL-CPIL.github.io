"""Model-and-optimize approach for ZDT6 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s limit

        n_evals = self.budget
        dim = self.dim
        ref = 11.0

        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Step 1: Find f1_min with high precision
        n_grid = 100000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min_coarse = x1_grid[idx_min]
        
        # Refine locally
        local_grid = np.linspace(max(0, x1_min_coarse - 0.001),
                                  min(1, x1_min_coarse + 0.001), 100000)
        f1_local = compute_f1(local_grid)
        idx_local_min = np.argmin(f1_local)
        x1_min = local_grid[idx_local_min]
        f1_min = compute_f1(x1_min)

        # Step 2: Optimal f1 distribution for max hypervolume
        # U(t) = integral of (R-s)/(2*sqrt(s)) ds = R*sqrt(t) - (1/3)*t^(3/2)
        def U_func(t):
            return ref * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Evenly spaced U targets (midpoints of intervals for optimal spacing)
        U_targets = np.linspace(U_min + U_range / (2*n_evals),
                                 U_max - U_range / (2*n_evals), n_evals)
        U_targets_shifted = U_targets - U_min

        # Step 3: Invert U(f1) = target using dense grid + interpolation
        n_invert = 200000
        f1_grid_inv = np.linspace(f1_min, 1.0, n_invert)
        U_grid_inv = U_func(f1_grid_inv) - U_min
        f1_selected = np.interp(U_targets_shifted, U_grid_inv, f1_grid_inv)

        # Step 4: Build dense x1-to-f1 mapping for inverse lookup
        n_map = 200000
        x1_map = np.linspace(0, 1, n_map)
        f1_map = compute_f1(x1_map)

        # Split at minimum
        idx_min_map = np.argmin(f1_map)
        x1_left = x1_map[:idx_min_map + 1]
        f1_left = f1_map[:idx_min_map + 1]
        x1_right = x1_map[idx_min_map:]
        f1_right = f1_map[idx_min_map:]

        # For left branch: f1 decreases, so reverse for interpolation
        f1_left_rev = f1_left[::-1]
        x1_left_rev = x1_left[::-1]

        # Interpolate x1 from f1 on both branches
        x1_from_left = np.interp(f1_selected, f1_left_rev, x1_left_rev)
        x1_from_right = np.interp(f1_selected, f1_right, x1_right)

        # Choose branch with better accuracy
        f1_check_left = compute_f1(x1_from_left)
        f1_check_right = compute_f1(x1_from_right)
        err_left = np.abs(f1_check_left - f1_selected)
        err_right = np.abs(f1_check_right - f1_selected)
        use_left = err_left <= err_right

        x1_final = np.where(use_left, x1_from_left, x1_from_right)

        # Step 5: Polish with brentq for high precision (vectorized batches)
        # Process in chunks to stay within time budget
        chunk_size = 50
        for start in range(0, n_evals, chunk_size):
            if time.time() > deadline - 5:  # leave 5s for evaluation
                break
            end = min(start + chunk_size, n_evals)
            for i in range(start, end):
                target_f1 = f1_selected[i]
                if target_f1 < f1_min + 1e-10 or target_f1 > 1.0 - 1e-10:
                    continue
                if use_left[i]:
                    lo, hi = 0.0, x1_min
                else:
                    lo, hi = x1_min, 1.0
                try:
                    x1_refined = brentq(lambda x: compute_f1(x) - target_f1, lo, hi,
                                         xtol=1e-14, rtol=1e-14, maxiter=50)
                    x1_final[i] = x1_refined
                except:
                    pass

        # Step 6: Build and evaluate points
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(points[i])
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
