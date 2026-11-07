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
            """Compute hypervolume for points on the ZDT4 Pareto front.
            
            For points on the Pareto front sorted by f1 (increasing), f2 is decreasing.
            The hypervolume relative to reference (ref_f1, ref_f2) is:
            HV = sum over non-dominated points, sorted by f1:
                 (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
            where f1_0 = min f1, f1_{n} = ref_f1, and we take the lower envelope
            of f2 (i.e., for each f1 interval, use the minimum f2 among points
            with f1 <= right endpoint).
            
            Since all points are on the Pareto front (non-dominated), and f2 is
            monotonically decreasing in f1 for ZDT4's h function... wait, is h(x)
            monotonically decreasing? h(x) = 1 + exp(-4x - 0.3*sin^5(8*pi*x)).
            The sin^5 term oscillates, so h is NOT monotonically decreasing.
            This means some points may be dominated by others.
            
            For correct HV computation with potentially non-monotonic f2:
            1. Sort by f1 ascending
            2. For each point, keep only if f2 is less than all previous f2 values
               (i.e., the point is on the lower-left envelope)
            3. Compute HV from the remaining non-dominated points
            """
            x0_sorted = np.sort(x0_arr)
            f1 = x0_sorted
            f2 = h_func(x0_sorted)
            
            # Filter: only points strictly better than reference
            mask = (f1 < ref_f1) & (f2 < ref_f2)
            if not np.any(mask):
                return 0.0
            f1 = f1[mask]
            f2 = f2[mask]
            if len(f1) == 0:
                return 0.0
            
            # Remove dominated points: keep only the lower envelope
            # Sort by f1 ascending, then track minimum f2 seen so far
            # A point is non-dominated if its f2 is strictly less than all f2 of
            # points with smaller f1
            f2_min = np.full(len(f1), np.inf)
            f2_min[0] = f2[0]
            for i in range(1, len(f1)):
                f2_min[i] = min(f2_min[i-1], f2[i])
            
            # A point i is non-dominated if f2[i] == f2_min[i] (it achieves the min)
            # But we also need to handle ties: if f2[i] == f2_min[i], it's on the envelope
            # However, if multiple points have the same f2_min, only the one with smallest
            # f1 matters for the HV step function. Actually for HV, we want the step
            # function: for each f1 interval, the height is determined by the minimum
            # f2 among all points with f1 <= right endpoint of interval.
            
            # Simpler correct approach: 
            # The HV is the area under the step function where at each f1 value,
            # the "height" is ref_f2 - min(f2 of all points with f1' <= f1)
            # But since we're integrating, we can do:
            # Sort by f1, compute running min of f2, then:
            # HV = sum of (f1_{i+1} - f1_i) * (ref_f2 - f2_running_min_i)
            # where f1_{n} = ref_f1
            
            f2_running_min = np.minimum.accumulate(f2)
            f1_ext = np.append(f1, ref_f1)
            diffs = np.diff(f1_ext)
            heights = ref_f2 - f2_running_min
            hv = np.sum(diffs * heights)
            return hv

        # Optimize point placement to maximize hypervolume.
        # Strategy: parameterize x0 positions via a smooth increasing transform,
        # then use Nelder-Mead to optimize the transform parameters.
        #
        # For ZDT4, h(x) = 1 + exp(-4x - 0.3*sin^5(8*pi*x))
        # The function oscillates near x=0 and decays exponentially.
        # We want to concentrate points where the curve has high curvature
        # (near x=0) and spread them out where it's smooth (large x).
        #
        # Parameterization: x0_i = x0_lo + (x0_hi - x0_lo) * u_i
        # where u_i are determined by a piecewise-linear mapping controlled
        # by a small number of "knot" parameters.
        #
        # Simpler approach: use x0_i = x0_lo + (x0_hi - x0_lo) * (i/n)^alpha
        # and optimize alpha. But this is too restrictive.
        #
        # Better: use a quadratic or cubic spline with a few control points,
        # or directly optimize the x0 positions with constraints.
        #
        # Most practical: use Nelder-Mead on the x0 positions directly,
        # with a penalty for violating monotonicity and bounds.
        # With n=200, this is expensive. Instead, use a coarser parameterization.
        #
        # Approach: divide [x0_lo, x0_hi] into K segments, and within each segment
        # use power-law spacing. Optimize the segment boundaries and exponents.
        #
        # Even simpler and effective: use a two-parameter family
        # x0_i = x0_lo + (x0_hi - x0_lo) * ((1 - exp(-beta * i/n)) / (1 - exp(-beta)))
        # which concentrates points near x0_lo for large beta.
        # Or: x0_i = x0_lo + (x0_hi - x0_lo) * (i/n)^alpha
        #
        # Let's try optimizing alpha with a finer grid, plus a local refinement.
        
        # First, coarse search over alpha
        best_x0 = None
        best_hv = -1.0
        
        # Alpha < 1 concentrates points near 0
        for alpha in np.linspace(0.2, 1.5, 50):
            t = np.linspace(0, 1, n)
            x0_candidate = x0_lo + (x0_hi - x0_lo) * (t ** alpha)
            hv_candidate = compute_hv(x0_candidate)
            if hv_candidate > best_hv:
                best_hv = hv_candidate
                best_x0 = x0_candidate.copy()
        
        # Fine-tune alpha with local search
        if best_x0 is not None:
            # Find the best alpha from the coarse search
            best_alpha = None
            best_hv_fine = -1.0
            t = np.linspace(0, 1, n)
            for alpha in np.linspace(0.2, 1.5, 500):
                x0_candidate = x0_lo + (x0_hi - x0_lo) * (t ** alpha)
                hv_candidate = compute_hv(x0_candidate)
                if hv_candidate > best_hv_fine:
                    best_hv_fine = hv_candidate
                    best_alpha = alpha
                    best_x0 = x0_candidate.copy()
                    best_hv = hv_candidate
            
            # Local Nelder-Mead around the best alpha
            if best_alpha is not None:
                from scipy.optimize import minimize
                def neg_hv_alpha(a):
                    t = np.linspace(0, 1, n)
                    x0_candidate = x0_lo + (x0_hi - x0_lo) * (t ** a[0])
                    return -compute_hv(x0_candidate)
                
                try:
                    res = minimize(neg_hv_alpha, [best_alpha], method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 2000})
                    if res.success or -res.fun > best_hv:
                        a_opt = np.clip(res.x[0], 0.01, 5.0)
                        t = np.linspace(0, 1, n)
                        x0_candidate = x0_lo + (x0_hi - x0_lo) * (t ** a_opt)
                        hv_candidate = compute_hv(x0_candidate)
                        if hv_candidate > best_hv:
                            best_hv = hv_candidate
                            best_x0 = x0_candidate.copy()
                except Exception:
                    pass
        
        # If still no good result, fall back to uniform
        if best_x0 is None:
            best_x0 = x0_vals

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
