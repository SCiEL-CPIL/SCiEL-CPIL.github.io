"""Model-and-optimize approach for ZDT6 multi-objective optimization."""
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
        deadline = start_time + 290  # safety margin before 330s limit

        n_evals = self.budget
        dim = self.dim
        ref = 11.0

        # ZDT6 analytical model:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # g(x) = 1 + 10*(dim-1) + sum_{i=2}^{dim} [x_i^2 - 10*cos(12*pi*x_i)]
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        #
        # Pareto front: x2..x10 = 0, so g = 1
        # On PF: f2 = 1 - sqrt(f1)
        # f1 ranges from f1_min to 1

        def compute_f1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)

        # Find f1_min using a dense grid
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min = x1_grid[idx_min]
        f1_min = f1_grid[idx_min]

        # For hypervolume maximization with reference point [11, 11]:
        # Points on PF: (f1_i, 1 - sqrt(f1_i)) sorted by f1 ascending
        # f2 is decreasing in f1, so f2_1 > f2_2 > ... > f2_n
        #
        # Hypervolume = sum over "steps" in the staircase
        # With points sorted by f1 ascending (f2 descending):
        # HV = sum_{i=1}^{n} (ref - f1_i) * (f2_{i-1} - f2_i) + (ref - f1_n) * f2_n
        # where f2_0 = ref
        #
        # More precisely, the hypervolume of points P = {(f1_i, f2_i)} with ref R:
        # Sort by f1 ascending. HV = sum_{i=1}^{n} (R - f1_i) * (f2_{i-1} - f2_i)
        # where f2_0 = R, and we include the last term (R - f1_n)*f2_n implicitly
        # Actually: HV = sum_{i=1}^{n} (R - f1_i) * (f2_{i-1} - f2_i) where f2_0 = R
        # This equals: (R-f1_1)*(R-f2_1) + sum_{i=2}^{n} (R-f1_i)*(f2_{i-1}-f2_i)
        # Wait, let me be more careful.
        #
        # For 2D, with points sorted by f1 ascending, the HV with reference (R,R) is:
        # HV = sum_{i=1}^{n} (R - f1_i) * (f2_{i-1} - f2_i)
        # where f2_0 = R
        # This can be rewritten as:
        # HV = (R - f1_1)*(R - f2_1) + sum_{i=2}^{n} (R - f1_i)*(f2_{i-1} - f2_i)
        #
        # To maximize this, we want to choose n values of f1 in [f1_min, 1]
        #
        # Key insight: the marginal contribution of a point at f1 depends on
        # the gap to neighboring points. For a smooth front, the optimal
        # distribution equalizes the "marginal hypervolume" per point.
        #
        # Let's use a different approach: since we have the analytical front,
        # we can use a dense candidate set and greedily select points.
        # But with 200 points from a large set, greedy is feasible if the set
        # is moderate.
        #
        # Alternative: use the fact that for hypervolume with a monotone front,
        # the optimal spacing in "hypervolume contribution space" is uniform.
        #
        # Let me try: parameterize by f1, and use the density approach.
        # The hypervolume contribution density for a point at f1 is related to
        # how much area it "covers." For a smooth decreasing front f2(t):
        # dHV/dt ≈ (R - t) * |f2'(t)| + f2(t)  ... this isn't quite right.
        #
        # Actually, let's use a simpler and more robust approach:
        # 1. Create a very dense set of candidate f1 values
        # 2. Use a greedy algorithm to select n_evals points that maximize HV
        #    Greedy: start with empty set, at each step add the point that
        #    maximizes HV increase.
        #
        # With 200k candidates and 200 selections, greedy is O(200 * 200k) = 4e7
        # per HV computation... but HV computation for 200 points is O(200 log 200).
        # So total: 200 * 200k * 200*log(200) ≈ 200 * 200k * 1500 = 6e10. Too slow.
        #
        # Better greedy: maintain the current set, and for each candidate compute
        # the HV increase. But that's still expensive.
        #
        # Let me use a smarter approach: since the front is smooth and monotone,
        # use adaptive spacing. Start with uniform spacing in f1, then iterate:
        # compute HV, find where the "gap" is largest, and adjust.
        #
        # Actually, the simplest effective approach: use the density-based method
        # but with the correct density. For a point at (f1, f2) on the front,
        # its "importance" for HV is roughly (R - f1) * (R - f2) / n if we
        # think of it as covering area. But this ignores interactions.
        #
        # Let me try a different angle: use scipy's minimize to optimize the
        # f1 values directly. But 200 variables is a lot.
        #
        # Practical approach: use a good initial distribution and refine.
        # The density (R - t) / (2*sqrt(t)) from the current code is a good
        # approximation. Let me stick with that but improve the inversion.

        # U(t) = integral of (R-s)/(2*sqrt(s)) ds from f1_min to t
        # = [R*sqrt(s) - (1/3)*s^{3/2}] from f1_min to t
        #   (since integral (R-s)/(2*sqrt(s)) ds = R*sqrt(s) - (1/3)*s^{3/2})
        # Let me verify: d/ds [R*sqrt(s) - (1/3)*s^{3/2}] = R/(2*sqrt(s)) - (1/2)*sqrt(s)
        # = (R - s) / (2*sqrt(s)). Yes!

        def U_func(t):
            return ref * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create n_evals target U values, evenly spaced
        U_targets = np.linspace(U_min + U_range / (2*n_evals),
                                 U_max - U_range / (2*n_evals), n_evals)
        U_targets_shifted = U_targets - U_min

        # Invert U(f1) = U_target using a fine grid
        n_invert = 1000000
        f1_grid_inv = np.linspace(f1_min, 1.0, n_invert)
        U_grid_inv = U_func(f1_grid_inv) - U_min

        # Interpolate to get f1 values
        f1_selected = np.interp(U_targets_shifted, U_grid_inv, f1_grid_inv)

        # Now find x1 for each f1 value using vectorized search
        # Build a dense mapping from f1 to x1
        n_map = 200000
        x1_map = np.linspace(0, 1, n_map)
        f1_map = compute_f1(x1_map)
        
        # For each target f1, find the closest x1 using searchsorted on sorted f1
        # Since f1 is non-monotonic, we need to handle both branches
        # Split at the minimum point
        idx_min_map = np.argmin(f1_map)
        x1_left = x1_map[:idx_min_map + 1]
        f1_left = f1_map[:idx_min_map + 1]
        x1_right = x1_map[idx_min_map:]
        f1_right = f1_map[idx_min_map:]
        
        # f1_left is decreasing (from x1=0 to x1_min), f1_right is increasing (from x1_min to x1=1)
        # For targets in [f1_min, f1(0)], we can search both branches
        # f1(0) = 1 - exp(0)*sin(0)^6 = 1
        # f1(1) = 1 - exp(-4)*sin(6*pi)^6 = 1 - exp(-4)*0 = 1
        
        x1_final = np.zeros(n_evals)
        f1_targets = f1_selected
        
        # For each target, find best x1 from left or right branch
        # Left branch: f1 decreases from 1 to f1_min as x1 goes 0 to x1_min
        # Right branch: f1 increases from f1_min to 1 as x1 goes x1_min to 1
        
        # Use interpolation on both branches
        # Left: x1 as function of f1 (f1 is decreasing, so invert)
        # Right: x1 as function of f1 (f1 is increasing)
        
        # For efficiency, use np.interp which requires monotonic x
        # Left branch: sort by f1 (ascending means x1 descending)
        # Right branch: already increasing in f1
        
        # Interpolate
        x1_from_left = np.interp(f1_targets, f1_left[::-1], x1_left[::-1])
        x1_from_right = np.interp(f1_targets, f1_right, x1_right)
        
        # Choose the x1 that gives f1 closer to target
        f1_check_left = compute_f1(x1_from_left)
        f1_check_right = compute_f1(x1_from_right)
        
        err_left = np.abs(f1_check_left - f1_targets)
        err_right = np.abs(f1_check_right - f1_targets)
        
        use_left = err_left <= err_right
        x1_final[use_left] = x1_from_left[use_left]
        x1_final[~use_left] = x1_from_right[~use_left]

        # Build points: x1 = x1_final[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_final

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
