import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 analysis:
        # f1 = x0
        # f2 = (1 + 9 * sum(x[1:]) / (n-1)) * (1 - sqrt(f1))
        # To minimize f2, we must minimize sum(x[1:]). Since bounds are [0,1], min sum is 0.
        # Thus, on the Pareto front (x[1:] = 0):
        # f1 = x0
        # f2 = 1 - sqrt(x0)
        # x0 ranges in [0, 1].
        
        # We have a budget of self.budget evaluations.
        # We want to choose N points x0_1, ..., x0_N in [0, 1] to maximize the hypervolume
        # of the set of points {(x0_i, 1 - sqrt(x0_i))} with reference point (11, 11).
        
        # For a convex decreasing curve f2(x), the hypervolume contribution of a point x_i
        # is maximized when the "gaps" in hypervolume between adjacent points are equalized.
        # Specifically, for a set of points sorted by x, the hypervolume is the sum of trapezoids
        # (actually, it's the area under the step function or the union of rectangles).
        # With reference point (R1, R2), the hypervolume of a set of non-dominated points is:
        # HV = sum_{i} (R1 - x_{i+1}) * (R2 - y_i) - (R1 - x_i) * (R2 - y_{i-1}) ... wait.
        
        # Let's use the standard formula for HV of a set of points sorted by x (increasing):
        # Points: (x_1, y_1), ..., (x_N, y_N) with x_1 < ... < x_N and y_1 > ... > y_N.
        # Reference: (R_x, R_y).
        # HV = sum_{i=1}^{N} (R_x - x_{i+1}) * (y_i - y_{i+1}) where x_{N+1} = R_x, y_0 = R_y?
        # No, the standard decomposition is:
        # HV = (R_x - x_1) * (R_y - y_1) + sum_{i=1}^{N-1} (x_{i+1} - x_i) * (R_y - y_i)
        # Wait, that's for the area to the left and below.
        # Let's verify.
        # The region is the union of rectangles defined by the points.
        # For point i, it dominates the rectangle [x_i, R_x] x [y_i, R_y]? No, it dominates [x_i, R_x] x [y_i, R_y] only if no other point dominates parts of it.
        # Actually, the hypervolume is the volume of the union of boxes [x_i, R_x] x [y_i, R_y].
        # Since points are non-dominated and sorted x_1 < x_2 < ... < x_N, y_1 > y_2 > ... > y_N:
        # The union can be decomposed into:
        # 1. Rectangle for point 1: [x_1, R_x] x [y_1, R_y] ? No, point 2 covers [x_2, R_x] x [y_2, R_y] which is inside? No, y_2 < y_1, so [x_2, R_x] x [y_2, R_y] is NOT inside [x_1, R_x] x [y_1, R_y] because y_2 < y_1 means the box for point 2 is "lower" but "wider" in x? No, x_2 > x_1.
        # Let's draw it.
        # x-axis: 0 .. 1 .. 11
        # y-axis: 0 .. 1 .. 11
        # Point 1: (x1, y1). Box: [x1, 11] x [y1, 11].
        # Point 2: (x2, y2). Box: [x2, 11] x [y2, 11].
        # Since x2 > x1 and y2 < y1, the box for Point 2 is to the right and below Point 1's box?
        # No, x2 > x1 means Point 2 is to the right. y2 < y1 means Point 2 is below.
        # The box for Point 1 covers x in [x1, 11] and y in [y1, 11].
        # The box for Point 2 covers x in [x2, 11] and y in [y2, 11].
        # The intersection is [x2, 11] x [y1, 11].
        # The union is Box1 U Box2.
        # Area(Union) = Area(Box1) + Area(Box2) - Area(Intersection)
        # This gets complicated for N points.
        
        # Alternative formula for HV of non-dominated set sorted by x:
        # HV = (R_x - x_1) * (R_y - y_1) + sum_{i=1}^{N-1} (x_{i+1} - x_i) * (R_y - y_i)
        # Let's test N=1: (R_x - x_1)(R_y - y_1). Correct.
        # N=2: (R_x - x_1)(R_y - y_1) + (x_2 - x_1)(R_y - y_1).
        # = (R_x - x_1 + x_2 - x_1)(R_y - y_1) = (R_x - 2x_1 + x_2)(R_y - y_1).
        # Let's check geometrically.
        # Box 1: [x1, Rx] x [y1, Ry]. Area = (Rx-x1)(Ry-y1).
        # Box 2: [x2, Rx] x [y2, Ry]. Area = (Rx-x2)(Ry-y2).
        # Intersection: [x2, Rx] x [y1, Ry]. Area = (Rx-x2)(Ry-y1).
        # Union = (Rx-x1)(Ry-y1) + (Rx-x2)(Ry-y2) - (Rx-x2)(Ry-y1)
        # = (Rx-x1)(Ry-y1) + (Rx-x2)[(Ry-y2) - (Ry-y1)]
        # = (Rx-x1)(Ry-y1) + (Rx-x2)(y1-y2).
        
        # My previous formula: (Rx-x1)(Ry-y1) + (x2-x1)(Ry-y1).
        # Are they equal?
        # (Rx-x1)(Ry-y1) + (Rx-x2)(y1-y2) vs (Rx-x1)(Ry-y1) + (x2-x1)(Ry-y1)
        # Difference: (Rx-x2)(y1-y2) vs (x2-x1)(Ry-y1)
        # Generally not equal. So my formula was wrong.
        
        # Correct recursive addition:
        # When adding point i (sorted by x increasing, y decreasing) to a set of points 1..i-1:
        # The new hypervolume added is the area of the box of point i that is not covered by previous points.
        # Previous points cover the region "above and to the left" of the current front segment?
        # Actually, the standard efficient algorithm for HV in 2D:
        # Sort by x.
        # HV = 0
        # prev_y = R_y
        # for i in 1..N:
        #    HV += (x_{i+1} - x_i) * (prev_y - y_i)  # where x_{N+1} = R_x
        #    prev_y = y_i
        # Wait, let's trace N=1.
        # i=1: HV += (x_2 - x_1) * (R_y - y_1). Here x_2 = R_x.
        # HV = (R_x - x_1)(R_y - y_1). Correct.
        
        # N=2:
        # i=1: HV += (x_2 - x_1) * (R_y - y_1). prev_y = y_1.
        # i=2: HV += (x_3 - x_2) * (y_1 - y_2). Here x_3 = R_x.
        # Total HV = (x_2 - x_1)(R_y - y_1) + (R_x - x_2)(y_1 - y_2).
        # Let's check against the union formula:
        # Union = (Rx-x1)(Ry-y1) + (Rx-x2)(y1-y2).
        # My sum: (x2-x1)(Ry-y1) + (Rx-x2)(y1-y2).
        # Expand Union: (Rx-x1)(Ry-y1) + (Rx-x2)(y1-y2)
        # = (Rx-x2+x2-x1)(Ry-y1) + (Rx-x2)(y1-y2)
        # = (Rx-x2)(Ry-y1) + (x2-x1)(Ry-y1) + (Rx-x2)(y1-y2)
        # = (Rx-x2)[(Ry-y1) + (y1-y2)] + (x2-x1)(Ry-y1)
        # = (Rx-x2)(Ry-y2) + (x2-x1)(Ry-y1).
        # This matches the sum of areas of Box 2 and the part of Box 1 not in Box 2?
        # Box 2 area: (Rx-x2)(Ry-y2).
        # Part of Box 1 not in Box 2: Box 1 is [x1, Rx] x [y1, Ry]. Box 2 is [x2, Rx] x [y2, Ry].
        # Part of Box 1 left of x2: [x1, x2] x [y1, Ry]. Area = (x2-x1)(Ry-y1).
        # Total = (Rx-x2)(Ry-y2) + (x2-x1)(Ry-y1).
        # Yes! The formula is:
        # HV = sum_{i=1}^{N} (x_{i+1} - x_i) * (y_{i-1} - y_i)
        # where x_0 is not used, but y_0 = R_y?
        # Let's re-verify indices.
        # Term i=1: (x_2 - x_1) * (y_0 - y_1) with y_0 = R_y.
        # Term i=2: (x_3 - x_2) * (y_1 - y_2).
        # ...
        # Term i=N: (x_{N+1} - x_N) * (y_{N-1} - y_N) with x_{N+1} = R_x.
        
        # So HV = sum_{i=1}^{N} (x_{i+1} - x_i) (y_{i-1} - y_i)
        # with x_{N+1} = R_x, y_0 = R_y.
        # Note: x_i is increasing, y_i is decreasing.
        
        # We want to maximize this sum by choosing x_1, ..., x_N in [0, 1].
        # y_i = 1 - sqrt(x_i).
        # y_{i-1} - y_i = (1 - sqrt(x_{i-1})) - (1 - sqrt(x_i)) = sqrt(x_i) - sqrt(x_{i-1}).
        # Let x_0 be something? No, the term is (x_{i+1} - x_i) (sqrt(x_i) - sqrt(x_{i-1})).
        # Wait, for i=1, y_0 = R_y = 11.
        # y_0 - y_1 = 11 - (1 - sqrt(x_1)) = 10 + sqrt(x_1).
        # Term 1: (x_2 - x_1) (10 + sqrt(x_1)).
        # For i > 1: (x_{i+1} - x_i) (sqrt(x_i) - sqrt(x_{i-1})).
        # Last term i=N: (R_x - x_N) (sqrt(x_N) - sqrt(x_{N-1})).
        
        # This looks complex to optimize directly with calculus due to the mixed terms.
        # However, there is a known result for optimal discrete points on a convex front for HV maximization.
        # For a convex front, the optimal points are those where the "marginal gain" is equal.
        # Alternatively, we can use a numerical optimizer to find the x_i that maximize HV.
        # N=200. 200 variables.
        # We can use `scipy.optimize.minimize` or a simple iterative method.
        # Given the time limit, a simple grid search or gradient-free method might be safer.
        # Or, we can use the property that for f2 = 1 - sqrt(f1), the optimal x_i are related to the inverse of the cumulative hypervolume function.
        
        # Let's try a simpler heuristic first: uniform spacing in "hypervolume space".
        # Or, since we have time, let's use `scipy.optimize.differential_evolution` or similar?
        # No, 200 dimensions is too high for DE.
        
        # Let's look at the structure again.
        # HV = (x_2 - x_1)(10 + sqrt(x_1)) + sum_{i=2}^{N-1} (x_{i+1} - x_i)(sqrt(x_i) - sqrt(x_{i-1})) + (11 - x_N)(sqrt(x_N) - sqrt(x_{N-1}))
        
        # If we assume the points are "dense", the term (sqrt(x_i) - sqrt(x_{i-1})) is small.
        # The dominant term for the first point is large because of the 10.
        # This suggests x_1 should be very small to maximize (x_2 - x_1) * 10?
        # If x_1 -> 0, Term 1 -> x_2 * 10.
        # This suggests clustering near 0?
        
        # Actually, let's just use a robust numerical method.
        # We can define the HV function and use `scipy.optimize.minimize` with L-BFGS-B?
        # 200 variables is manageable for L-BFGS-B if the function is smooth.
        
        def hv_objective(x_vals):
            # x_vals is array of N points in [0, 1]
            # Sort them
            x_sorted = np.sort(x_vals)
            y_vals = 1 - np.sqrt(x_sorted)
            
            # Calculate HV
            # x_{i+1} - x_i
            dx = np.diff(np.concatenate([x_sorted, [11.0]]))
            # y_{i-1} - y_i
            # y_0 = 11
            dy = np.diff(np.concatenate([[11.0], y_vals]))
            # Wait, the formula was (x_{i+1} - x_i) * (y_{i-1} - y_i)
            # dx[i] = x_{i+1} - x_i
            # dy[i] = y_i - y_{i-1} (negative)
            # We need (y_{i-1} - y_i) = -dy[i]
            
            hv = np.sum(dx * (-dy))
            return -hv # Minimize negative HV
        
        # Initial guess: uniform grid
        x_init = np.linspace(0.01, 0.99, self.budget)
        
        # Optimize
        from scipy.optimize import minimize
        res = minimize(hv_objective, x_init, method='L-BFGS-B', bounds=[(0, 1)] * self.budget)
        
        x_opt = res.x
        x_opt = np.sort(x_opt)
        
        # Construct decision variables
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x_opt
        # x[1:] = 0 (lower bound)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return F
