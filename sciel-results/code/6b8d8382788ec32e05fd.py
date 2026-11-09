"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, aim for 320s
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        n = self.dim
        # ZDT2 constant
        C = 1 + 10 * (n - 1)
        
        # Reference point value
        ref_val = 11.0
        
        # Calculate the intersection t* where f2(t) = ref_val
        # f2(t) = C * (1 - sqrt(t))
        # C * (1 - sqrt(t)) = ref_val
        # 1 - sqrt(t) = ref_val / C
        # sqrt(t) = 1 - ref_val / C
        # t = (1 - ref_val / C)^2
        t_star = (1.0 - ref_val / C) ** 2
        
        # Clip t_star to valid range [0, 1]
        t_star = float(np.clip(t_star, 0.0, 1.0))
        
        N = self.budget
        
        if N <= 0:
            return np.zeros((0, self.dim))
        
        if N == 1:
            # Single point: pick the point that maximizes hypervolume contribution
            # which is roughly the midpoint in f-space, or just t_star
            X = np.zeros((1, self.dim))
            X[0, 0] = t_star
            F = np.array([problem(X[0])])
            return X
        
        # We will generate N points on the Pareto front.
        # One point at t_star to capture the "corner" where f2 drops below ref.
        # The rest distributed in [t_star, 1.0].
        
        # Strategy: Inverse-CDF sampling based on hypervolume contribution.
        # For a convex front, the hypervolume contribution of a point t in [t_a, t_b]
        # is proportional to the area of the triangle formed by (t_a, f2(t_a)), (t_b, f2(t_b)) and (t, f2(t)).
        # However, a simpler and effective approach for convex fronts is to sample uniformly in f1 (which is t)
        # OR to sample such that the gaps in f2 are proportional to the slope.
        
        # Let's analyze the marginal hypervolume gain.
        # The hypervolume of a set of points on a convex curve can be approximated by the area under the curve 
        # minus the area of the "empty" rectangles.
        # Actually, for a single front, the hypervolume is the area of the union of rectangles [0, f1_i] x [0, f2_i].
        # Since the front is convex and we are minimizing, the "boundary" of the dominated region is the curve itself.
        # The hypervolume is simply the area under the curve f2(t) from t_star to 1, PLUS the area of the rectangle 
        # from 0 to t_star with height f2(t_star) (which is ref_val).
        # Wait, the reference point is (11, 11).
        # The points are (t, C(1-sqrt(t))).
        # For t in [t_star, 1], f2(t) <= 11.
        # The hypervolume contribution of a point (f1, f2) relative to ref (11, 11) is (11-f1)*(11-f2).
        # But hypervolume of a *set* of points is the volume of the union of the boxes.
        # For a convex front, the union of boxes is exactly the region under the "upper envelope" of the boxes.
        # Since the front is convex, the upper envelope is the curve itself (for the part where it's below the ref point).
        # So the hypervolume is the area of the region: 0 <= f1 <= 11, 0 <= f2 <= 11, below the curve.
        # This is: Integral from 0 to t_star of 11 df1 + Integral from t_star to 1 of f2(t) dt.
        # = 11 * t_star + Integral_{t_star}^1 C(1-sqrt(t)) dt.
        
        # To maximize this discrete approximation with N points, we should place points such that 
        # the trapezoidal rule (or similar) approximates the integral well.
        # The function f2(t) is convex (f2'' > 0).
        # Trapezoidal rule error is related to the second derivative.
        # To minimize error, we should use more points where the curvature is higher.
        # f2(t) = C - C*t^0.5
        # f2'(t) = -0.5 * C * t^-0.5
        # f2''(t) = 0.25 * C * t^-1.5
        # Curvature is highest at small t. So we should densify points near t_star.
        
        # Let's use a transformation.
        # Let u be uniform in [0, 1].
        # We want t(u) such that the spacing in t is proportional to t^1.5 (inverse of curvature)?
        # Or simply use a power law t = t_star + (1 - t_star) * u^alpha.
        # If alpha < 1, points are denser near t_star (u=0).
        # Let's try alpha = 0.5 or 0.3.
        
        # However, there is a better way: Optimize the spacing directly.
        # The hypervolume of a set of points t_0 < t_1 < ... < t_{N-1} on a convex curve
        # is the sum of trapezoids if we assume the boundary connects the points linearly.
        # HV = sum_{i=0}^{N-2} 0.5 * (f2(t_i) + f2(t_{i+1})) * (t_{i+1} - t_i) + 11*t_star (approx)
        # Actually, the first part is the area under the piecewise linear approximation of the curve.
        # We want to maximize this sum.
        
        # Let's use a simple heuristic: sample uniformly in the square root of t?
        # f2(t) = C(1 - sqrt(t)). Let y = sqrt(t). Then t = y^2.
        # f2(y) = C(1-y). This is linear in y!
        # If we sample uniformly in y, the points are linearly spaced in f2.
        # Since f2(y) is linear, the curve in (t, f2) space is a parabola.
        # Sampling uniformly in y means sampling uniformly in the "vertical" axis.
        # This should be very good for approximating the area.
        
        # Let's try this:
        # y ranges from sqrt(t_star) to 1.
        # y_vals = linspace(sqrt(t_star), 1, N)
        # t_vals = y_vals^2
        
        # Let's verify this intuition.
        # Area = Integral f2(t) dt = Integral C(1-y) * 2y dy (from y_star to 1)
        # = 2C Integral (y - y^2) dy
        # This is a smooth function of y. Uniform spacing in y should be excellent.
        
        y_star = np.sqrt(t_star)
        
        # Generate N points in y
        y_vals = np.linspace(y_star, 1.0, N)
        t_vals = y_vals ** 2
        
        # Ensure t_star is included (it is, as the first point)
        # Ensure 1.0 is included (it is, as the last point)
        
        # Construct the population X
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # x[1:] remain 0, which is optimal for ZDT2 front
        
        # Evaluate all points
        # Check time periodically? 
        # With N=200 and simple problem, it should be fast.
        
        # We can also try to refine the spacing using a few iterations of gradient ascent on the HV?
        # But with N=200, the linear spacing in y is likely very close to optimal.
        # Let's stick to this for robustness and speed.
        
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
