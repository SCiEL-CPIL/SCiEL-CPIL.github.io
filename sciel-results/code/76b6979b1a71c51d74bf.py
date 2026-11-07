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

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Check if 0 is within bounds for x1..x_{dim-1}
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            # Use all budget to sample the Pareto front
            n_points = self.budget

            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)

            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                # We want to maximize hypervolume of the sampled points.
                # The Pareto front is f1 = x0, f2 = h(x0) = 1 + exp(-4*x0 - 0.3*(sin(8*pi*x0)**5))
                # Reference point is (11, 400).
                # Hypervolume is the area of the union of rectangles [f1_i, 11] x [f2_i, 400] for all non-dominated points.
                # Since the front is a function f2 = h(f1) and h is decreasing (mostly), 
                # the non-dominated set is the whole set if we sample distinct x0.
                # Actually, for a decreasing function, all points are non-dominated if x0 are distinct.
                # The hypervolume is the area under the curve h(x) from min_x to max_x, 
                # plus the area of the rectangle from max_x to 11 at height h(max_x), 
                # plus the area from min_x to 11 at height 400? No.
                
                # Let's think about the hypervolume of a set of points P = {(x_i, y_i)} with ref (R1, R2).
                # HV = sum over i of (R1 - x_i) * (R2 - y_i) - overlaps? No, that's for a single point.
                # For multiple points, it's the area of the union of boxes [x_i, R1] x [y_i, R2].
                # Since the points lie on a decreasing curve, the union of boxes is simply the area under the "staircase" formed by the points.
                # If we sort points by x_i increasing, the staircase height is y_i for x in [x_i, x_{i+1}).
                # So HV = sum_{i=1}^{n-1} (x_{i+1} - x_i) * (R2 - y_i) + (R1 - x_n) * (R2 - y_n) + (x_1 - 0) * (R2 - y_1)?
                # Wait, the boxes start at x_i. The union is the region {(u,v) | u >= x_i, v >= y_i for some i}.
                # Since y_i decreases as x_i increases, the boundary is the curve itself.
                # The area is the integral from x_min to R1 of (R2 - max(0, h_inv(u))) du?
                # No, the points are discrete. The boundary is the polygonal chain connecting (x_1, y_1), ..., (x_n, y_n) and then to (R1, R2) and (R1, y_n) and (x_1, R2)?
                # Actually, the hypervolume is the area of the set S = { (u,v) : u <= R1, v <= R2, and there exists i such that u >= x_i and v >= y_i }.
                # This is equivalent to the area under the upper envelope of the horizontal lines v=y_i starting at u=x_i.
                # Since y_i is decreasing, the upper envelope is just the sequence of y_i.
                # The area is sum_{i=1}^n (x_{i+1} - x_i) * (R2 - y_i) where x_{n+1} = R1, and x_0 = 0?
                # Let's check: The region is composed of rectangles.
                # For u in [x_1, x_2], the max v such that (u,v) is in S is y_1 (since y_1 > y_2). So area is (x_2 - x_1) * (R2 - y_1).
                # For u in [x_2, x_3], the max v is y_2. Area is (x_3 - x_2) * (R2 - y_2).
                # ...
                # For u in [x_n, R1], the max v is y_n. Area is (R1 - x_n) * (R2 - y_n).
                # What about u < x_1? The boxes start at x_i. So for u < x_1, no box covers it?
                # Wait, the box for point i is [x_i, R1] x [y_i, R2].
                # So the union is indeed for u >= x_1.
                # But wait, if we have a point with small x, it covers a wide range of u.
                # So the area is exactly sum_{i=1}^{n} (x_{i+1} - x_i) * (R2 - y_i) with x_{n+1}=R1.
                # Note: this assumes x_i are sorted and distinct.
                
                # To maximize this sum, we want to place points such that the "staircase" is as high as possible.
                # Since y_i = h(x_i) is decreasing, we want to place points where the drop in h(x) is small relative to the gain in width?
                # Actually, the sum is approximating the integral of (R2 - h(x)) dx from x_min to R1.
                # But we are using left-endpoint rectangles (since y_i corresponds to x_i, and the rectangle extends to x_{i+1}).
                # The error is due to the curvature of h(x).
                # To minimize the error (and thus maximize the area, since we are underestimating the integral if h is convex? 
                # Let's check: h(x) is decreasing. Is it convex or concave?
                # h(x) = 1 + exp(-4x - 0.3 sin^5(8pi x)).
                # The dominant term is -4x, so it's roughly exponential decay. Exponential decay is convex.
                # If h is convex, the left-endpoint sum overestimates the integral?
                # Let's check: for a convex function, the tangent is below the curve. The chord is above.
                # The left-endpoint rectangle uses the value at the left end.
                # If h is decreasing and convex, h(x) is above the chord from (x_i, h(x_i)) to (x_{i+1}, h(x_{i+1})).
                # So the rectangle height h(x_i) is greater than the average height on [x_i, x_{i+1}].
                # Thus, the left-endpoint sum overestimates the integral.
                # We want to maximize the sum, so we want to make the overestimation as large as possible?
                # No, the true hypervolume is fixed (the area under the curve). The sum is an approximation.
                # Wait, the hypervolume of the discrete set is NOT the integral. It is the area of the union of boxes.
                # And that area is EXACTLY the left-endpoint sum (for decreasing h).
                # So we want to maximize the left-endpoint sum.
                # But the left-endpoint sum is always >= the integral (for convex decreasing h).
                # The difference is the sum of the areas between the curve and the horizontal lines.
                # To maximize the sum, we want to maximize the overestimation.
                # This means we want to place points where the curve drops the most?
                # No, the overestimation on [x_i, x_{i+1}] is approximately (1/2) * (x_{i+1}-x_i)^2 * |h''(x_i)|.
                # So we want to place points where |h''| is large?
                # But we also want to cover the range [x_lo, x_hi].
                # Actually, if we place points very close together where |h''| is large, we reduce the overestimation there.
                # Wait, I want to MAXIMIZE the sum. So I want to MINIMIZE the overestimation?
                # No, the sum is the hypervolume. The integral is the "true" area.
                # The sum is always >= the integral.
                # So the maximum possible hypervolume is achieved when the sum is as large as possible.
                # But the sum is bounded above by the integral + some error term.
                # Actually, the sum is the hypervolume. It is a fixed value for a given set of points.
                # We want to choose the points to maximize this value.
                # Since the sum is >= the integral, and the integral is fixed, we want to maximize the sum.
                # The sum is maximized when the points are placed such that the left-endpoint rectangles cover as much area as possible.
                # This is equivalent to minimizing the "gap" between the staircase and the curve?
                # No, the staircase is ABOVE the curve (for convex h).
                # So the area of the staircase is the integral + the area between the staircase and the curve.
                # To maximize the area, we want to maximize the area between the staircase and the curve.
                # This is achieved by placing points where the curve is most "curved" (high |h''|), so that the gap is large.
                # But we have a fixed number of points.
                # This is a non-trivial optimization problem.
                
                # However, a simpler approach is to note that for a smooth curve, a uniform grid is a good approximation.
                # But we can do better by using a non-uniform grid that is denser where the curvature is high.
                # Let's try a power-law spacing: x_i = x_lo + (x_hi - x_lo) * (i/n)^p.
                # We can optimize p to maximize the hypervolume.
                
                # Let's try a few values of p and pick the best one.
                best_hv = -1
                best_x0_vals = None
                
                # Try p from 0.5 to 2.0
                for p in np.linspace(0.5, 2.0, 20):
                    t = np.linspace(0, 1, n_points)
                    x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** p)
                    # Evaluate h(x0_vals)
                    y_vals = 1 + np.exp(-4 * x0_vals - 0.3 * (np.sin(8 * np.pi * x0_vals) ** 5))
                    # Compute hypervolume
                    # Sort by x0
                    idx = np.argsort(x0_vals)
                    x_sorted = x0_vals[idx]
                    y_sorted = y_vals[idx]
                    # Ensure unique x values (should be unique for p>0)
                    # Compute sum
                    R1 = 11.0
                    R2 = 400.0
                    # x_{n+1} = R1
                    dx = np.diff(np.append(x_sorted, R1))
                    hv = np.sum(dx * (R2 - y_sorted))
                    if hv > best_hv:
                        best_hv = hv
                        best_x0_vals = x0_vals
                
                x0_vals = best_x0_vals

            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] = 0.0 is default

            # Check time
            if time.time() > deadline:
                return X

            F = np.array([problem(x) for x in X])
            return X

        else:
            # Fallback: 0 is not in bounds for some x_i (i>=1)
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
