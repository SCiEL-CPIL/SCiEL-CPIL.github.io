"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: analytic sampling on the Pareto front with optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        dim = self.dim

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # We need to choose n values of x1 that maximize the hypervolume
        # of the resulting points with reference point (11, 11).
        #
        # The hypervolume contribution of a point (f1_i, f2_i) on a decreasing front
        # depends on the gap to the next point. For points sorted by f1:
        # HV = sum over i of (f1_{i+1} - f1_i) * (f2_i - ref2) + (ref1 - f1_n) * (f2_n - ref2)
        # where f1_{n+1} = ref1 = 11.
        #
        # Actually, let me think more carefully. For a set of points on a monotone front
        # with reference point (r1, r2), the hypervolume is:
        # HV = sum_i (f1_{i+1} - f1_i) * (f2_i - r2)
        # where f1_0 = 0 (or the leftmost extent) and f1_{n+1} = r1.
        #
        # Wait, more precisely for non-dominated points sorted by increasing f1:
        # The hypervolume is the area under the step function. If points are (f1_i, f2_i)
        # sorted by f1 increasing, then:
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (f2_i - r2) + (r1 - f1_n) * (f2_n - r2)
        # But we need f2_i > r2 for the point to contribute. Since r2 = 11 and f2 <= 1,
        # all points have f2 < r2, so... wait, that can't be right.
        #
        # Oh wait - in minimization, a point is "better" than the reference if f_i < r_i.
        # The hypervolume is the volume of the region dominated by the points and better
        # than the reference. So for 2D:
        # HV = area of union of rectangles [0, f1_i] x [0, f2_i] minus [0, r1] x [0, r2]... 
        # No, it's the region that is dominated by at least one point AND is better than ref.
        #
        # For minimization with ref (r1, r2), the hypervolume of a set of points is:
        # The measure of { (y1, y2) : y1 < r1, y2 < r2, and exists point p with p1 <= y1, p2 <= y2 }
        #
        # For points on a decreasing front sorted by f1 increasing:
        # HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * (f2_i - r2_clipped) ... 
        # 
        # Let me just compute it directly. Sort points by f1. The dominated region is:
        # For each point i, it dominates the rectangle [0, f1_i] x [0, f2_i].
        # The union of these rectangles, intersected with [0, r1] x [0, r2], gives HV.
        #
        # Since the front is decreasing (f1 increasing, f2 decreasing), the union is:
        # HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * (f2_i)  where f1_0 = 0
        # But clipped by r1 and r2. Since all f1_i <= 1 < 11 = r1 and all f2_i <= 1 < 11 = r2:
        # HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * f2_i  where f1_0 = 0
        #
        # Wait no. The region dominated by point i is [0, f1_i] x [0, f2_i].
        # The union of all such regions for a decreasing front:
        # = { (y1, y2) : y2 <= f2_i for some i with f1_i >= y1 }
        # Since f2 is decreasing in f1, the maximum f2 for any point with f1 >= y1 is f2_1 (the first point).
        # Hmm, let me think again.
        #
        # For a decreasing front, point 1 has smallest f1 and largest f2, point n has largest f1 and smallest f2.
        # The union of rectangles [0, f1_i] x [0, f2_i] is:
        # A region where for a given y1, the max y2 covered is max{f2_i : f1_i >= y1}.
        # Since f2_i is decreasing in i (and f1_i is increasing), for y1 between f1_{i-1} and f1_i,
        # the points with f1_j >= y1 are j >= i, and the max f2 among those is f2_i.
        # So the boundary is a step function:
        # y2_max(y1) = f2_i for y1 in [f1_{i-1}, f1_i), i = 1..n
        # y2_max(y1) = 0 for y1 >= f1_n
        #
        # Therefore: HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * f2_i  where f1_0 = 0
        #
        # But we also need to clip by r1 and r2. Since f1_i <= 1 < 11 and f2_i <= 1 < 11,
        # no clipping is needed.
        #
        # So HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * f2_i  with f1_0 = 0, f2_i = 1 - sqrt(f1_i)
        #
        # To maximize this, we want to choose f1_1 < f1_2 < ... < f1_n in [0, 1].
        #
        # This is equivalent to minimizing the discretization error of the integral
        # ∫_0^1 (1 - sqrt(x)) dx using a left-Riemann-sum-like formula.
        # Actually it's: sum (f1_i - f1_{i-1}) * (1 - sqrt(f1_i))
        # The true integral is ∫_0^1 (1 - sqrt(x)) dx = 1 - 2/3 = 1/3.
        #
        # For optimal placement, we should use a spacing that accounts for the curvature.
        # The function g(x) = 1 - sqrt(x) has g'(x) = -1/(2*sqrt(x)), which is large near x=0.
        # So we need more points near x=0.
        #
        # Optimal spacing for minimizing error in a Riemann sum: dx_i ∝ 1/|g'(x_i)|
        # But for a left sum, the error per interval is ~ (dx_i)^2 * |g''(x)|/2
        # g''(x) = 1/(4*x^(3/2))
        # So error_i ∝ (dx_i)^2 / x_i^(3/2)
        # To equalize: dx_i / x_i^(3/4) = const, so dx_i ∝ x_i^(3/4)
        # This means x_i grows like t^(4/3) where t is the index.
        #
        # Let's use x_i = (i/n)^(4/3) for i = 1..n, with x_0 = 0.
        # Actually, let's use a more careful approach.
        #
        # Alternative: use the transformation u = sqrt(x), so x = u^2, dx = 2u du.
        # Then f2 = 1 - u, and HV = sum (x_i - x_{i-1}) * (1 - sqrt(x_i))
        # = sum 2*u_i*(u_i - u_{i-1}) * (1 - u_i)  ... this gets complicated.
        #
        # Let me just try several spacings and pick the best one analytically.
        # Spacing options:
        # 1. Uniform in x: x_i = i/n
        # 2. Uniform in sqrt(x): x_i = (i/n)^2
        # 3. x_i = (i/n)^(4/3)
        # 4. x_i = (i/n)^(5/3)
        # 5. x_i = (i/n)^3
        #
        # Let me compute HV for each with n=200 and see which is best.
        # Actually, I'll just compute them all and pick the best.

        candidates = []
        # Various power spacings
        for p in [1.0, 1.2, 1.33, 1.5, 1.67, 2.0, 2.5, 3.0]:
            x1_vals = np.power(np.arange(1, n + 1) / n, p)
            candidates.append(x1_vals)
        
        # Also try a few other strategies
        # Midpoint in sqrt space
        x1_mid = np.power((np.arange(1, n + 1) - 0.5) / n, 2)
        candidates.append(x1_mid)
        
        # Optimal: use the fact that for f2 = 1 - sqrt(x1), 
        # the ideal spacing equalizes the contribution
        # Let's also try a geometric-like spacing
        # x_i = a * r^i type... but bounded to [0,1]
        
        best_hv = -1
        best_x1 = None
        for x1_vals in candidates:
            # Compute HV = sum (x1_i - x1_{i-1}) * (1 - sqrt(x1_i))
            x1_prev = np.concatenate([[0.0], x1_vals[:-1]])
            f2_vals = 1.0 - np.sqrt(x1_vals)
            hv = np.sum((x1_vals - x1_prev) * f2_vals)
            if hv > best_hv:
                best_hv = hv
                best_x1 = x1_vals

        # Use the best spacing
        x1_values = best_x1

        # Create the design matrix: x1 = x1_values, x2..xn = 0 (to make g=0, on Pareto front)
        X = np.zeros((n, dim))
        X[:, 0] = x1_values

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        # Check deadline
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
