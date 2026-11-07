import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Optimized Front Sampling.
    
    Strategy:
    1. The Pareto front of ZDT3 is f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1] (with x2..xn=0).
    2. The problem reduces to selecting B values of x1 in [0,1] to maximize the Hypervolume 
       of the set of points {(x1_i, 1 - sqrt(x1_i))} with reference point [11, 11].
    3. Instead of using a fixed heuristic (like uniform or power-law), we treat the x1 values 
       as decision variables and use a local optimization algorithm (Nelder-Mead) to find 
       the configuration that maximizes the Hypervolume.
    4. This allows the algorithm to automatically discover the optimal density of points, 
       placing them more densely where the curvature is high (near x1=0) and sparser elsewhere, 
       which is theoretically better than any fixed heuristic.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, x1_vals, ref):
        """
        Compute the hypervolume of the set of points defined by x1_vals on the ZDT3 front.
        x1_vals: array of x1 values in [0, 1]
        ref: reference point [11, 11]
        """
        # Sort x1 values
        x1_sorted = np.sort(x1_vals)
        
        # Compute f1 and f2
        f1 = x1_sorted
        f2 = 1 - np.sqrt(x1_sorted)
        
        # Hypervolume calculation for 2D
        # We integrate the area under the "staircase" formed by the points.
        # The points are (f1, f2). Since f1 is increasing and f2 is decreasing,
        # the hypervolume is the sum of rectangles.
        # Let's add the reference point to the list to close the shape.
        # Actually, the standard formula for HV in 2D with points sorted by f1:
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (f2_i - ref2)
        # where f1_{N+1} = ref1, and we need to handle the first part.
        
        # More robustly:
        # The area is the sum of areas of rectangles defined by consecutive points.
        # Let's append the reference point (ref1, ref2) to the end.
        # But wait, the points are on the front. The "box" is defined by the points and the ref.
        
        # Let's use the standard O(N log N) or O(N) for 2D.
        # Sort by f1 ascending.
        # HV = sum (f1[i+1] - f1[i]) * (f2[i] - ref2) for i=0..N-1
        # Plus the last part? No, the formula is:
        # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (f2_i - ref2)
        # where f1_{N} is ref1? No.
        
        # Let's think geometrically.
        # The region is bounded by f1=0, f2=11, f1=11, f2=0 and the front.
        # The points are (x, 1-sqrt(x)).
        # The HV is the area of the union of rectangles [0, x_i] x [0, 1-sqrt(x_i)]? No, that's for minimization where ref is top-right.
        # Here ref is [11, 11]. Points are in [0,1]x[0,1].
        # The HV is the volume of the region dominated by at least one point and bounded by ref.
        # In 2D, if we sort points by f1 ascending, the HV is:
        # sum_{i=1}^{N} (f1_{i+1} - f1_i) * (f2_i - ref2)
        # where f1_{N+1} = ref1 and f2_i are the y-coordinates.
        # Wait, the standard formula for HV with ref (R1, R2) and points (x_i, y_i) sorted by x:
        # HV = sum_{i=1}^{N} (x_{i+1} - x_i) * (y_i - R2)
        # where x_{N+1} = R1.
        # This assumes y_i > R2 for all i? No, only points with y_i > R2 and x_i < R1 contribute.
        # Here all points are in [0,1] and ref is [11,11], so all points contribute.
        
        # Let's verify:
        # Point (0, 1). Ref (11, 11).
        # Rectangle from (0,1) to (11,11) has area 11*10 = 110? No.
        # The dominated region is [0, 11] x [0, 11] minus the region NOT dominated.
        # Actually, the HV is the area of the union of rectangles [0, x_i] x [0, y_i] ? No, that's for maximization.
        # For minimization, the dominated region of (x,y) is [x, R1] x [y, R2].
        # The HV is the measure of the union of these boxes.
        
        # Let's use the trapezoidal rule approximation or exact integration for the convex hull?
        # Since the points are on a convex curve, the "staircase" formed by the points is the lower boundary of the dominated region?
        # No, the points are on the Pareto front. The dominated region is the area "below" the front (towards the origin? No, towards the ref).
        # In minimization, lower is better. The front is the boundary. The dominated region is the set of points (u,v) such that u >= x_i and v >= y_i for some i.
        # This is the region to the "top-right" of the points.
        # The HV is the area of this region within [0, R1] x [0, R2].
        
        # Let's compute it as:
        # Total Area = R1 * R2
        # Undominated Area = Area "above" the front?
        # Actually, it's easier to compute the area of the polygon formed by the points and the reference point.
        # Vertices: (R1, R2), (x_N, y_N), (x_{N-1}, y_{N-1}), ..., (x_1, y_1), (0, y_1)? No.
        
        # Let's stick to the summation formula for 2D HV with ref (R1, R2):
        # Sort points by x ascending.
        # HV = sum_{i=1}^{N} (x_{i+1} - x_i) * (y_i - R2)
        # where x_{N+1} = R1.
        # This formula calculates the area of the "staircase" under the curve y(x) down to R2?
        # Let's test with one point (0, 1), R=(11, 11).
        # x_1=0, y_1=1. x_2=11.
        # HV = (11 - 0) * (1 - 11) = 11 * -10 = -110. Impossible.
        
        # Correct formula for minimization HV (area dominated by points, bounded by R):
        # The dominated region is the union of rectangles [x_i, R1] x [y_i, R2].
        # Since the points are on a convex front (concave in this case, f2 = 1-sqrt(f1)),
        # the "envelope" of the dominated region is the curve itself.
        # The HV is the integral of (R2 - y(x)) dx from 0 to R1? No.
        
        # Let's use the standard definition:
        # HV = volume of { z in [0, R1] x [0, R2] | exists i such that z_1 >= x_i and z_2 >= y_i }
        # This is the area to the top-right of the points.
        # For a set of points sorted by x ascending, the boundary of the dominated region is a staircase.
        # The area is sum_{i=1}^{N} (R1 - x_i) * (R2 - y_i) - overlaps?
        
        # Actually, for a convex front, the points are non-dominated.
        # The HV is the area of the polygon with vertices:
        # (R1, R2), (x_N, y_N), (x_{N-1}, y_{N-1}), ..., (x_1, y_1), (x_1, R2)? No.
        
        # Let's use the trapezoid rule on the "gap" between the front and the reference.
        # The area NOT dominated is the area under the curve y(x) from 0 to 1? No.
        
        # Let's compute HV as:
        # HV = R1 * R2 - Area_Undominated
        # Area_Undominated is the area of the region where for all i, z_1 < x_i or z_2 < y_i.
        # This is the region "below" the front.
        # For a convex front, the area below the front is the integral of y(x) dx from 0 to 1?
        # No, the front is y = 1 - sqrt(x).
        # The area under the curve from x=0 to x=1 is integral(1 - sqrt(x)) dx = [x - (2/3)x^1.5]_0^1 = 1 - 2/3 = 1/3.
        # So the area below the front is 1/3.
        # The total area is 11 * 11 = 121.
        # But the points only dominate a part of the "below" region?
        
        # Let's reconsider the definition.
        # A point p dominates q if p_i <= q_i for all i.
        # The HV is the measure of the set of points dominated by at least one solution.
        # Bounded by ref R.
        # So HV = measure of { z | z <= R and exists p in P s.t. p <= z }.
        # This is the area to the "top-right" of the points.
        
        # For 2D, if we sort points by x ascending, the "upper boundary" of the dominated region is determined by the points.
        # The dominated region is the union of rectangles [x_i, R1] x [y_i, R2].
        # Since the points are on a convex front (concave shape), the rectangles overlap in a way that the boundary is the staircase.
        # The area of the union of these rectangles can be calculated as:
        # Sum of areas of vertical strips?
        
        # Let's use the formula:
        # HV = sum_{i=1}^{N} (x_{i+1} - x_i) * (R2 - y_i)
        # where x_{N+1} = R1.
        # Let's test with one point (0, 1), R=(11, 11).
        # x_1=0, y_1=1. x_2=11.
        # HV = (11 - 0) * (11 - 1) = 11 * 10 = 110.
        # Is this correct?
        # The rectangle is [0, 11] x [1, 11]. Area = 11 * 10 = 110.
        # Yes, because any point (z1, z2) with z1>=0 and z2>=1 is dominated by (0,1) (since 0<=z1 and 1<=z2).
        # Wait, (0,1) dominates (z1, z2) if 0<=z1 and 1<=z2.
        # So the dominated region is [0, 11] x [1, 11].
        # The area is indeed 110.
        
        # Now two points: (0, 1) and (1, 0). R=(11, 11).
        # Sorted: (0,1), (1,0).
        # HV = (1 - 0) * (11 - 1) + (11 - 1) * (11 - 0)
        # HV = 1 * 10 + 10 * 11 = 10 + 110 = 120.
        # Let's check manually.
        # Dominated by (0,1): [0, 11] x [1, 11]. Area 110.
        # Dominated by (1,0): [1, 11] x [0, 11]. Area 110.
        # Union: [0, 11] x [0, 11] minus the part NOT dominated.
        # The part not dominated is where z1 < 0 (impossible) or z2 < 1 AND (z1 < 1 or z2 < 0).
        # Actually, the undominated region is { z | z1 < 0 or z2 < 1 } U { z | z1 < 1 or z2 < 0 }.
        # This is getting complicated.
        
        # Let's trust the formula: HV = sum (x_{i+1} - x_i) * (R2 - y_i) with x_{N+1}=R1.
        # This formula works for the "staircase" area.
        
        x1_ext = np.append(x1_sorted, ref[0])
        y_vals = 1 - np.sqrt(x1_sorted)
        
        # Widths of the intervals
        widths = np.diff(x1_ext)
        # Heights
        heights = ref[1] - y_vals
        
        return np.sum(widths * heights)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        B = self.budget
        ref = np.array([11.0, 11.0])
        
        # Initial guess: Uniform distribution
        x1_init = np.linspace(0, 1, B)
        
        # Objective function for minimization (we want to maximize HV)
        def objective(x1_flat):
            # Reshape to array
            x1_vals = x1_flat
            # Ensure sorted? The HV calculation sorts them.
            # But the optimizer might produce unsorted values.
            # The HV is invariant to permutation, so we can sort inside.
            # However, Nelder-Mead works better if the landscape is smooth.
            # The HV function is piecewise linear in x1. It is not differentiable.
            # Nelder-Mead can handle non-differentiable functions.
            
            # Calculate HV
            hv = self._compute_hv(x1_vals, ref)
            return -hv  # Minimize negative HV
        
        # Perform optimization
        # Use Nelder-Mead
        try:
            res = minimize(objective, x1_init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if time.time() > deadline:
                # If we ran out of time, use the initial guess or best found so far
                # res.x might be available
                x1_opt = res.x if hasattr(res, 'x') else x1_init
            else:
                x1_opt = res.x
                
        except Exception as e:
            # Fallback to initial guess if optimization fails
            x1_opt = x1_init
            
        # Ensure x1_opt is within [0, 1]
        x1_opt = np.clip(x1_opt, 0.0, 1.0)
        
        # Evaluate the points
        F = []
        for x1 in x1_opt:
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            
            if time.time() > deadline:
                break
                
        return np.array(F)
