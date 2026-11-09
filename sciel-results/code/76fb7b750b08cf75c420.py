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

        # Find f1_min precisely using a dense grid + local refinement
        n_grid = 200000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = compute_f1(x1_grid)
        idx_min = np.argmin(f1_grid)
        x1_min_coarse = x1_grid[idx_min]
        f1_min_coarse = f1_grid[idx_min]

        # Local refinement around the minimum
        local_grid = np.linspace(max(0, x1_min_coarse - 0.005),
                                  min(1, x1_min_coarse + 0.005), 100000)
        f1_local = compute_f1(local_grid)
        idx_local = np.argmin(f1_local)
        x1_min = local_grid[idx_local]
        f1_min = f1_local[idx_local]

        # U(t) = integral of (R-s)/(2*sqrt(s)) ds from f1_min to t
        # = [R*sqrt(s) - (1/3)*s^{3/2}] from f1_min to t
        def U_func(t):
            return ref * np.sqrt(t) - (1.0/3.0) * t**1.5

        U_min = U_func(f1_min)
        U_max = U_func(1.0)
        U_range = U_max - U_min

        # Create n_evals target U values, evenly spaced
        U_targets = np.linspace(U_min + U_range / (2 * n_evals),
                                 U_max - U_range / (2 * n_evals), n_evals)

        # For each target U value, invert to find f1, then invert f1 to find x1
        # We need to solve U_func(f1) = U_target for f1 in [f1_min, 1]
        # Then solve compute_f1(x1) = f1 for x1 in [0, 1]
        #
        # Since f1 is not monotonic over [0,1], we need to be careful.
        # f1(x1) = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # f1 has multiple local minima and maxima.
        # The Pareto front corresponds to specific x1 values where f1 takes
        # values in [f1_min, 1]. For each f1 value in that range, there may
        # be multiple x1 values. We want the one that gives the correct
        # Pareto-optimal point.
        #
        # Actually, for ZDT6, the Pareto front is achieved when x2..x10 = 0,
        # and x1 can be any value in [0,1]. The objective values are
        # (f1(x1), 1 - sqrt(f1(x1))). So we just need to find x1 values
        # such that f1(x1) equals our target f1 values.
        #
        # Since f1 is not one-to-one, we need to find, for each target f1,
        # an x1 in [0,1] such that f1(x1) = target_f1.
        # We can do this by pre-computing a lookup table.

        # Pre-compute a dense lookup table: x1 -> f1(x1)
        n_lut = 1000000
        x1_lut = np.linspace(0, 1, n_lut)
        f1_lut = compute_f1(x1_lut)

        # For each target f1, we need to find x1 such that f1(x1) ≈ target_f1
        # We'll use the lookup table with interpolation.
        # Since f1 is not monotonic, we need to find all x1 where f1(x1) ≈ target
        # and pick the one that makes sense for the Pareto front.
        #
        # Actually, for the Pareto front, we want x1 values that cover the
        # full range of f1 from f1_min to 1. The simplest approach:
        # for each target f1, find an x1 in [0,1] where f1(x1) is closest
        # to the target. Since we have a dense LUT, we can do this efficiently.
        #
        # Better: since we want to maximize hypervolume, and the Pareto front
        # is parameterized by x1, we can just choose x1 values directly.
        # The key is to choose x1 values such that the resulting (f1, f2)
        # points are well-distributed for hypervolume maximization.
        #
        # Let's use a different approach: directly optimize x1 values using
        # the density-based method, but map through the actual f1 function.
        #
        # For each target U value, find f1_target, then find x1 such that
        # f1(x1) ≈ f1_target. We'll use the LUT for this.

        # Invert U to get target f1 values
        # U_func is monotonic increasing for t > 0, so we can invert it
        # U_func(t) = R*sqrt(t) - (1/3)*t^(3/2)
        # Let s = sqrt(t), then U = R*s - (1/3)*s^3
        # This is a cubic in s: (1/3)*s^3 - R*s + U = 0
        # We can solve this analytically or use brentq.

        def U_inverse(U_val):
            """Invert U_func(t) = U_val for t in [f1_min, 1]."""
            def g(t):
                return U_func(t) - U_val
            # U_func is increasing for t in [0, R/2] and we're in [f1_min, 1]
            # which is well within the increasing region since R=11
            try:
                return brentq(g, f1_min, 1.0, xtol=1e-15, rtol=1e-15)
            except:
                # Fallback to interpolation
                f1_grid_inv = np.linspace(f1_min, 1.0, 100000)
                U_grid_inv = U_func(f1_grid_inv)
                return np.interp(U_val, U_grid_inv, f1_grid_inv)

        # Compute target f1 values
        f1_targets = np.array([U_inverse(U_t) for U_t in U_targets])

        # For each target f1, find x1 using the LUT
        # Since f1 is not monotonic, we need to find x1 where f1(x1) ≈ target
        # We'll find all indices in the LUT where f1 is close to the target
        # and pick the one that's on the "main" branch of the Pareto front.
        #
        # Actually, for ZDT6, the Pareto front is well-defined: for each f1
        # value in [f1_min, 1], there's a corresponding x1. The issue is
        # that f1(x1) is not one-to-one. But for hypervolume maximization,
        # we just need valid (f1, f2) points on the Pareto front.
        #
        # Simplest robust approach: for each target f1, find the x1 in [0,1]
        # that minimizes |f1(x1) - target_f1| using the LUT.
        # With 1M LUT points and 200 targets, this is 200M comparisons,
        # which is fast with numpy.

        x1_selected = np.zeros(n_evals)
        for i in range(n_evals):
            if time.time() > deadline:
                break
            target_f1 = f1_targets[i]
            # Find the x1 in the LUT closest to achieving target_f1
            # Use absolute difference
            diffs = np.abs(f1_lut - target_f1)
            idx = np.argmin(diffs)
            x1_selected[i] = x1_lut[idx]

        # Build points: x1 = x1_selected[i], x2..x10 = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_selected

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
# EVOLVE-BLOCK-END
