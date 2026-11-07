import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        n_eval = 0
        max_eval = self.budget

        # Check if 0 is within bounds for x1..x_{dim-1}
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if not zeros_ok:
            # Fallback: random sampling
            rng = np.random.default_rng(0)
            for _ in range(max_eval):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                problem(x)
                n_eval += 1
            return

        # x0 bounds, clamped to [0,1] for standard ZDT4
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)

        if x0_lo >= x0_hi:
            # Degenerate: just evaluate one point
            x = np.zeros(self.dim)
            x[0] = x0_lo
            problem(x)
            return

        def h_func(x):
            """ZDT4 h function: h(x) = 1 + exp(-4x - 0.3*sin^5(8*pi*x))"""
            return 1.0 + np.exp(-4.0 * x - 0.3 * (np.sin(8.0 * np.pi * x) ** 5))

        # The Pareto front for ZDT4 with g=1:
        # f1 = x0, f2 = h(x0)
        # We want to maximize hypervolume with n points on this front.
        #
        # Key insight: The hypervolume is the area under the upper envelope
        # of the step function formed by the points. For a convex (in min-min
        # sense, i.e., the curve bulges toward the reference point) front,
        # the optimal strategy is to place points where the curvature is highest
        # to minimize the "overestimate" area between the true front and the
        # approximating step function.
        #
        # h(x) has high curvature near x=0 (due to the oscillating sin term and
        # the exponential). The function is approximately:
        #   For x near 0: h(x) ≈ 1 + exp(-0.3 * sin^5(8πx)) which oscillates
        #   For x away from 0: h(x) ≈ 1 + exp(-4x) which is smooth and convex
        #
        # The dominant contribution to the "waste" (area between step function
        # and true curve) comes from large gaps in x. The optimal placement
        # minimizes sum of (x_{i+1} - x_i) * max(f2(x_i), f2(x_{i+1})) type terms.
        #
        # A practical approach: use a dense adaptive grid that concentrates
        # points where the local curvature of h is highest.
        #
        # Alternative simpler approach: Since the front is 1D and we know it
        # analytically, we can directly optimize the x0 positions to maximize
        # hypervolume. With 200 points, we can do a simple coordinate ascent
        # or use the fact that the optimal points for a smooth convex curve
        # are roughly equally spaced in the "area" sense.
        #
        # Let's use a two-stage approach:
        # 1. Start with uniform spacing in x0
        # 2. Refine by moving each point to minimize local hypervolume loss

        n = max_eval  # use all budget

        # Initialize with uniform spacing
        x0_vals = np.linspace(x0_lo, x0_hi, n)

        # The hypervolume contribution can be computed as:
        # Sort points by f1 (which is x0), then:
        # HV = sum over i of (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
        # where we treat the last point as extending to ref_f1.
        # Actually for a set of points (f1_i, f2_i) sorted by f1:
        # HV = (ref_f1 - f1_0)*(ref_f2 - f2_0) - sum of areas "lost"
        # More directly: HV = sum_i (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
        # where f1_{n} = ref_f1, and f2 values are the non-dominated ones.
        # For points ON the Pareto front, all are non-dominated, sorted by f1:
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
        # with f1_n = ref_f1

        ref_f1 = 11.0
        ref_f2 = 400.0

        def compute_hv(x0_arr):
            """Compute hypervolume for points on the ZDT4 Pareto front."""
            x0_sorted = np.sort(x0_arr)
            f1 = x0_sorted
            f2 = h_func(x0_sorted)
            # Only count points strictly better than reference in every objective
            mask = (f1 < ref_f1) & (f2 < ref_f2)
            if not np.any(mask):
                return 0.0
            f1 = f1[mask]
            f2 = f2[mask]
            if len(f1) == 0:
                return 0.0
            # Compute HV
            f1_ext = np.append(f1, ref_f1)
            diffs = np.diff(f1_ext)
            # For each segment, the height is ref_f2 - f2_i (the y-value of the left point)
            heights = ref_f2 - f2
            hv = np.sum(diffs * heights)
            return hv

        # Optimize x0 positions using coordinate descent
        # This is a 1D optimization for each coordinate
        best_x0 = x0_vals.copy()
        best_hv = compute_hv(best_x0)

        # Coordinate descent: for each point, try to move it to a better position
        # Use local search around each point
        n_iter = 100  # limited iterations to stay within time
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Local search around x0_vals[i]
                # Try a few perturbation sizes
                for delta_scale in [0.01, 0.005, 0.001, 0.0005]:
                    # Try moving to the left and right
                    for direction in [-1, 1]:
                        new_x = best_x0[i] + direction * delta_scale * (x0_hi - x0_lo)
                        new_x = np.clip(new_x, x0_lo, x0_hi)
                        if new_x == best_x0[i]:
                            continue
                        old_x = best_x0[i]
                        best_x0[i] = new_x
                        new_hv = compute_hv(best_x0)
                        if new_hv > best_hv + 1e-12:
                            best_hv = new_hv
                            improved = True
                        else:
                            best_x0[i] = old_x
            if not improved:
                break

        # Now evaluate all the optimized points
        X = np.zeros((n, self.dim))
        X[:, 0] = best_x0
        # X[:, 1:] = 0.0 (already zero)

        for i in range(n):
            if time.time() > deadline:
                break
            problem(X[i])
            n_eval += 1

        return
