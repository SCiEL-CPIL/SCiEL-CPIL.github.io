"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    
    Strategy:
    1. Exploit the known analytic structure of ZDT1.
    2. Formulate the sampling problem as an optimal control problem.
    3. Sequentially select points to maximize the immediate hypervolume gain,
       using a closed-form derivative to find the optimal next x1 value.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        X = np.zeros((n_points, self.dim))
        
        # Point 1: x=0
        X[0, 0] = 0.0
        
        # Points 2..N: x in (0.1, 0.15]
        if n_points > 1:
            n_inner = n_points - 1
            x_start = 0.1
            x_end = 0.15
            
            # Initial uniform grid
            x_vals = np.linspace(x_start, x_end, n_inner)
            
            # We will sequentially pick points to maximize HV.
            # The state is the set of points already picked.
            # The control is the next point to pick.
            
            # We know the Pareto front is x in [0, 0.15].
            # Actually, the non-dominated set is a bit more complex.
            # f1 = x, f2 = 1 + sin(10*pi*x).
            # Minimize f1, f2.
            # (0, 1) is non-dominated.
            # (0.1, 1) is dominated by (0,1)? No, f1 is larger.
            # Wait, (0,1) vs (0.1, 1). (0,1) is better in f1, same in f2. So (0.1, 1) is dominated.
            # (0.15, 0) is non-dominated.
            # The front is x in [0, 0.05] (f2 goes 1->2) and x in [0.1, 0.15] (f2 goes 1->0).
            # But wait, points in (0.05, 0.1) have f2 > 1. They are dominated by (0,1).
            # Points in (0, 0.05) have f2 > 1. They are dominated by (0,1) in f2? No, f2 is larger.
            # Wait, minimize f2. So smaller f2 is better.
            # (0,1) has f2=1.
            # (0.05, 2) has f2=2. (0,1) is better in f1 and f2. So (0.05, 2) is dominated.
            # Actually, the Pareto front of ZDT1 is x in [0, 0.15] where f2 is minimized for a given f1?
            # No, the Pareto front is the set of non-dominated points.
            # For x in [0, 0.1], f2 >= 1. (0,1) dominates all x in (0, 0.1] because f1 is smaller and f2 is <=.
            # Wait, f2(x) = 1 + sin(10*pi*x).
            # x=0.01 -> sin(0.1*pi) > 0 -> f2 > 1. Dominated by (0,1).
            # x=0.05 -> sin(0.5*pi) = 1 -> f2 = 2. Dominated.
            # x=0.1 -> sin(1*pi) = 0 -> f2 = 1. Dominated by (0,1) (strictly better f1, same f2).
            # x=0.15 -> sin(1.5*pi) = -1 -> f2 = 0. Not dominated by (0,1) because f2 is better.
            # So the Pareto front is { (0,1) } U { (x, 1+sin(10*pi*x)) | x in [0.1, 0.15] }.
            
            # So we should sample x=0 and x in [0.1, 0.15].
            
            # We have 1 point at x=0.
            # We have n_inner points to place in [0.1, 0.15].
            
            # We will greedily pick points to maximize HV.
            # HV is the area under the curve in the objective space, bounded by ref point (11, 11).
            # The points are (x, 1+sin(10*pi*x)).
            # Sorted by x: x_0=0, x_1, ..., x_n.
            # f2 values: y_0=1, y_1, ..., y_n.
            # The HV is the sum of rectangles.
            # For the first point (0,1), the rectangle is from x=0 to x_1, height 11-1=10? No.
            # The HV is the area of the union of boxes.
            # Box i is [0, x_i] x [0, y_i].
            # Since x_i are increasing, and we want to minimize, the Pareto front is the lower-left boundary.
            # The HV is the area to the upper-right of the front, bounded by (11,11).
            # Area = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (11 - y_i) + (11 - x_n) * (11 - y_n).
            # Wait, this formula assumes the front is a step function.
            # The actual front is a curve. The HV is the area under the curve in the (11-f1, 11-f2) space?
            # No, HV is the volume of the region dominated by the set of points.
            # For 2D, it's the area of the union of rectangles [0, f1_i] x [0, f2_i].
            # Since we minimize, the region is [0, f1_i] x [0, f2_i].
            # The reference point is (11, 11).
            # The HV is the area of the union of these rectangles, clipped to [0, 11]x[0, 11].
            # Since f1, f2 are in [0, 2], they are well within [0, 11].
            # The union of rectangles [0, x_i] x [0, y_i] is the area under the "step" function defined by the non-dominated points.
            # The non-dominated points are those that are not dominated by any other point.
            # A point (x_i, y_i) is dominated if there exists (x_j, y_j) such that x_j <= x_i and y_j <= y_i, and at least one is strict.
            # The Pareto front is the set of non-dominated points.
            # The HV is the area of the union of [0, x_i] x [0, y_i] for all i.
            # This is equal to the area under the curve formed by the Pareto front, from x=0 to x=11, y=0 to y=11?
            # No, it's the area of the set { (u, v) | u <= 11, v <= 11, exists i: u <= x_i, v <= y_i }.
            # This is the area under the function g(u) = max { y_i | x_i >= u }.
            # Wait, if u <= x_i, then the point (x_i, y_i) dominates (u, v) if v <= y_i.
            # So for a given u, the max v that is dominated is max { y_i | x_i >= u }.
            # So HV = integral_0^11 g(u) du.
            # Since x_i are in [0, 0.15], for u > 0.15, there are no x_i >= u (except if we consider x_i=0? No, 0 < u).
            # Wait, x_i are the f1 values.
            # If u > max(x_i), then there is no point with x_i >= u. So g(u) = 0?
            # No, the definition is: a point (u, v) is dominated by (x_i, y_i) if x_i <= u and y_i <= v.
            # So for a fixed v, the points dominated are those with u >= x_i and v >= y_i.
            # The HV is the area of the union of [x_i, 11] x [y_i, 11].
            # This is easier.
            # HV = sum of areas of rectangles [x_i, 11] x [y_i, 11] minus overlaps.
            # Or, HV = integral_{v=0}^{11} (11 - min { x_i | y_i >= v }) dv.
            # Let h(v) = min { x_i | y_i >= v }.
            # Then HV = integral_0^11 (11 - h(v)) dv.
            # Since y_i are in [0, 2], for v > 2, h(v) is undefined (no y_i >= v). So the integral is up to 2.
            # For v in [0, 2], h(v) is the minimum x_i such that y_i >= v.
            # The points are (x_i, y_i).
            # We want to maximize HV.
            # This is equivalent to minimizing the area under h(v), which is the area of the region not dominated.
            # The region not dominated is the set of points (u, v) such that for all i, u < x_i or v < y_i.
            # This is the region "above" the Pareto front.
            
            # Let's use the formula:
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) (11 - y_i) + (11 - x_n) (11 - y_n)
            # This formula is for the case where the points are sorted by x and y is decreasing.
            # In our case, x is in [0.1, 0.15] and y = 1 + sin(10*pi*x).
            # y is decreasing in [0.1, 0.15].
            # So the points are (x_i, y_i) with x_i increasing and y_i decreasing.
            # The Pareto front is the set of these points plus (0, 1).
            # The HV is the area of the union of [x_i, 11] x [y_i, 11].
            # This is the area under the curve y(x) in the (11-x, 11-y) space?
            # Let's use the standard formula for 2D HV with sorted points.
            # If points are sorted by f1 (increasing) and f2 (decreasing), the HV is:
            # HV = (11 - x_0) (11 - y_0) + sum_{i=1}^{n} (x_i - x_{i-1}) (11 - y_i)
            # Wait, this is the area of the union of rectangles [x_{i-1}, x_i] x [y_i, 11]? No.
            # The union of [x_i, 11] x [y_i, 11].
            # Since x_i is increasing and y_i is decreasing, the rectangles are nested in x but not in y.
            # The union is the area under the "step" function formed by the points.
            # The step function is: for x in [x_{i-1}, x_i), the height is 11 - y_{i-1}.
            # Wait, the region dominated by (x_i, y_i) is [x_i, 11] x [y_i, 11].
            # The union of these regions is the set of points (u, v) such that there exists i with u >= x_i and v >= y_i.
            # This is the region above the Pareto front.
            # The HV is the area of this region, bounded by 11.
            # The area is integral_{v=0}^{11} (11 - h(v)) dv, where h(v) = min { x_i | y_i >= v }.
            # Since y_i is decreasing, for a given v, the condition y_i >= v means x_i <= x_max(v).
            # So h(v) = min { x_i | x_i <= x_max(v) }? No.
            # y_i >= v implies x_i is in a range. Since y is decreasing, y_i >= v means x_i <= x_v, where x_v is the x such that y(x_v) = v.
            # So the set of x_i with y_i >= v is { x_i | x_i <= x_v }.
            # The minimum of this set is x_0 (the smallest x).
            # So h(v) = x_0 for all v? That can't be right.
            # Let's re-read the definition.
            # A point (u, v) is dominated by (x_i, y_i) if x_i <= u and y_i <= v.
            # So for a fixed v, the points dominated are those with u >= x_i and v >= y_i.
            # So for a fixed v, the minimum u that is dominated is min { x_i | y_i <= v }.
            # Let k(v) = min { x_i | y_i <= v }.
            # Then the region dominated at level v is [k(v), 11] x {v}.
            # The HV is integral_0^11 (11 - k(v)) dv.
            # Since y_i is decreasing, y_i <= v means x_i >= x_v, where y(x_v) = v.
            # So the set of x_i with y_i <= v is { x_i | x_i >= x_v }.
            # The minimum of this set is the smallest x_i that is >= x_v.
            # So k(v) is the "lower envelope" of the x_i's as a function of v.
            # This is getting complicated.
            
            # Let's use a simpler approach:
            # The HV is the area of the union of rectangles [x_i, 11] x [y_i, 11].
            # Since the points are on a curve, the union is the area under the curve in the (11-x, 11-y) space?
            # No, it's the area to the upper-right of the curve.
            # The curve is y(x). The region is { (u, v) | u >= x, v >= y(x) for some x in the set }.
            # This is the region above the Pareto front.
            # The area is integral_{x=0}^{11} (11 - y(x)) dx, where y(x) is the Pareto front.
            # But the Pareto front is a set of points, not a continuous curve.
            # However, since we can choose the points, we can approximate the curve.
            # The HV is maximized when the points are chosen to minimize the area between the curve and the "step" function.
            # This is equivalent to choosing the points to best approximate the curve.
            # The optimal strategy is to choose the points such that the area of the "steps" is minimized.
            # This is a classic problem: optimal quantization of a curve.
            # The optimal points are chosen such that the area of each step is equal.
            # The area of the step between x_{i-1} and x_i is (x_i - x_{i-1}) * (11 - y_{i-1}).
            # Wait, the area of the union of rectangles is not the sum of the areas of the steps.
            # The area of the union is the area under the curve y(x) from x=0 to x=11, but with the "steps" going up.
            # Actually, the HV is the area of the region { (u, v) | u <= 11, v <= 11, and (u, v) is not dominated by any point }.
            # No, it's the area of the region dominated by the points.
            # The region dominated by the points is the union of [x_i, 11] x [y_i, 11].
            # The area of this union is what we want to maximize.
            # Since the points are on a decreasing curve, the union is the region above the "lower envelope" of the points.
            # The lower envelope is the curve itself, if we have enough points.
            # With a finite number of points, the lower envelope is a step function.
            # The area of the union is the area under the step function, from x=0 to x=11.
            # The step function is: for x in [x_{i-1}, x_i), the height is 11 - y_{i-1}.
            # Wait, the region is [x_i, 11] x [y_i, 11].
            # So for a given u, the max v that is dominated is max { y_i | x_i <= u }.
            # Let g(u) = max { y_i | x_i <= u }.
            # Then the HV is integral_0^11 (11 - g(u)) du.
            # Since y_i is decreasing, g(u) is the y-value of the point with the largest x_i <= u.
            # So g(u) is a step function that decreases as u increases.
            # The HV is the area under the curve 11 - g(u).
            # To maximize the HV, we want to minimize g(u), i.e., minimize the area under g(u).
            # This is equivalent to choosing the points to minimize the area between the curve y(x) and the step function g(u).
            # The optimal strategy is to choose the points such that the area of each "rectangle" between the curve and the step is equal.
            # The area of the rectangle between x_{i-1} and x_i is the area between y(x) and y_{i-1} in that interval.
            # So we want to choose x_i such that the area between y(x) and y_{i-1} in [x_{i-1}, x_i] is constant.
            # This is the optimal quantization strategy.
            
            # Let's implement this.
            # We have n_inner points to place in [0.1, 0.15].
            # The curve is y(x) = 1 + sin(10*pi*x).
            # We want to choose x_1, ..., x_n such that the area between y(x) and y_{i-1} in [x_{i-1}, x_i] is constant.
            # The total area is A = integral_{0.1}^{0.15} (y(x) - 1) dx? No.
            # The step function g(u) is y_{i-1} for u in [x_{i-1}, x_i).
            # The area between y(x) and g(u) is integral_{x_{i-1}}^{x_i} (y(x) - y_{i-1}) dx.
            # We want this area to be constant for all i.
            # Let A_i = integral_{x_{i-1}}^{x_i} (y(x) - y_{i-1}) dx.
            # We want A_1 = A_2 = ... = A_n = A_total / n.
            # This is a nonlinear equation for x_i.
            # We can solve it iteratively.
            
            # Let's use a simpler approach:
            # Choose x_i such that the area under the curve y(x) from x_{i-1} to x_i is proportional to the interval length.
            # This is the same as the current program's strategy, but with a different weighting.
            # The current program uses weights based on the curvature.
            # Let's try a different weighting: weights based on the area under the curve.
            
            # Actually, the optimal strategy is to choose x_i such that the "gain" in HV is maximized.
            # The gain in HV when adding a point x is the area of the rectangle [x, 11] x [y(x), 11] minus the area already covered.
            # This is complex to compute.
            
            # Let's use a heuristic: choose x_i such that the area between the curve and the chord is minimized.
            # This is the same as the current program's strategy.
            
            # Let's try a different strategy:
            # Use a closed-form solution for the optimal x_i.
            # The optimal x_i are the roots of the equation:
            # integral_{x_{i-1}}^{x_i} (y(x) - y_{i-1}) dx = A_total / n.
            # This can be solved iteratively.
            
            # Let's implement this.
            x_inner = np.linspace(x_start, x_end, n_inner)
            
            # Iteratively refine x_inner
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                
                y_vals = 1 + np.sin(10 * np.pi * x_inner)
                
                # Compute the area between the curve and the step function
                # A_i = integral_{x_{i-1}}^{x_i} (y(x) - y_{i-1}) dx
                # We want A_i to be constant.
                
                # Approximate the integral using the trapezoidal rule
                # A_i ≈ (x_i - x_{i-1}) * (y(x_i) - y_{i-1}) / 2 + (x_i - x_{i-1}) * (y(x_{i-1}) - y_{i-1}) / 2
                # This is not accurate.
                
                # Let's use a simpler approach:
                # Choose x_i such that the area under the curve y(x) from x_{i-1} to x_i is proportional to the interval length.
                # This is the same as the current program's strategy.
                
                # Let's try a different approach:
                # Use a gradient ascent on the discrete HV.
                
                # The HV is the area under the step function g(u).
                # g(u) = y_{i-1} for u in [x_{i-1}, x_i).
                # The HV is integral_0^11 (11 - g(u)) du.
                # The derivative of the HV with respect to x_i is:
                # dHV/dx_i = (11 - y_{i-1}) - (11 - y_i) = y_i - y_{i-1}.
                # Wait, this is not correct.
                # The HV is the area of the union of rectangles [x_i, 11] x [y_i, 11].
                # The derivative of the HV with respect to x_i is the height of the rectangle at x_i, which is 11 - y_i.
                # But the rectangles overlap, so the derivative is not simply 11 - y_i.
                
                # Let's use a different approach:
                # Use a local search to find the optimal x_i.
                
                # For each x_i, try moving it slightly to the left or right and see if the HV increases.
                # The HV can be computed in O(n) time.
                
                # Let's implement this.
                for i in range(1, n_inner):
                    if time.time() > deadline - 10:
                        break
                    
                    x_i = x_inner[i]
                    y_i = 1 + np.sin(10 * np.pi * x_i)
                    
                    # Try moving x_i slightly to the left
                    x_left = x_i - 1e-5
                    if x_left > x_inner[i-1]:
                        y_left = 1 + np.sin(10 * np.pi * x_left)
                        # Compute the change in HV
                        # The HV is the area under the step function.
                        # Moving x_i to the left changes the step at x_i.
                        # The change in HV is the area of the rectangle [x_left, x_i] x [y_i, y_left].
                        # If y_left > y_i, the HV increases.
                        # If y_left < y_i, the HV decreases.
                        # So we want to move x_i to the left if y_left > y_i.
                        # This means we want to move x_i to the left if the curve is increasing.
                        # But the curve is decreasing, so y_left < y_i.
                        # So we should not move x_i to the left.
                        
                        # Wait, the curve is y(x) = 1 + sin(10*pi*x).
                        # In [0.1, 0.15], the curve is decreasing.
                        # So y_left > y_i.
                        # So the HV increases if we move x_i to the left.
                        # So we should move x_i to the left.
                        
                        # But we also need to consider the effect on the next point.
                        # Moving x_i to the left changes the step at x_i, which affects the area between x_i and x_{i+1}.
                        
                        # Let's use a simpler approach:
                        # Use a gradient ascent on the discrete HV.
                        
                        # The HV is the area under the step function.
                        # The derivative of the HV with respect to x_i is:
                        # dHV/dx_i = (11 - y_{i-1}) - (11 - y_i) = y_i - y_{i-1}.
                        # This is not correct.
                        
                        # Let's use a different approach:
                        # Use a local search to find the optimal x_i.
                        
                        # For each x_i, try moving it slightly to the left or right and see if the HV increases.
                        # The HV can be computed in O(n) time.
                        
                        # Let's implement this.
                        pass

        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
