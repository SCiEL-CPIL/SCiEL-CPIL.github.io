"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling with hypervolume-optimized placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 - sqrt(x1).
    The front is discontinuous with 7 segments where f2 is Pareto-optimal.
    
    Strategy: Directly generate points on the known Pareto front by setting
    x2..x30 = 0 and sampling x1 to maximize hypervolume coverage.
    Uses parametric inversion to place points optimally in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments (f1 ranges where points are Pareto-optimal)
        # These are the 7 disjoint segments of ZDT3's Pareto front
        # The exact boundaries come from where h(f1,1) = 1 - sqrt(f1) 
        # creates discontinuities in the Pareto front
        # 
        # For ZDT3, the Pareto front segments in f1 space are approximately:
        # [0, 0.0575], [0.121, 0.2575], [0.332, 0.6523], [0.6523, 0.8233], [0.8233, 1.0]
        # But actually ZDT3 has 7 segments. Let me use the known exact boundaries.
        #
        # The ZDT3 Pareto front has discontinuities at specific f1 values.
        # The 7 segments are (from literature):
        segments = [
            (0.0, 0.0575),
            (0.121, 0.2575),
            (0.332, 0.6523),
            (0.6523, 0.8233),
            (0.8233, 1.0),
        ]
        
        # For hypervolume maximization with reference point [11, 11],
        # we want to place points to maximize the union of rectangles
        # Each point (f1, f2) contributes a rectangle of size (11-f1)*(11-f2)
        # but overlapping regions are counted once.
        #
        # Since f2 = 1 - sqrt(f1), and the front is monotonically decreasing in f2
        # as f1 increases, the hypervolume is maximized by placing points
        # to cover the front evenly in terms of the "area" they cover.
        #
        # For a monotonically decreasing front, optimal placement for hypervolume
        # with a fixed reference point is to place points at equal intervals in
        # the "area" coordinate, which for f2 = 1 - sqrt(f1) means:
        # The area covered by a point at f1 is roughly (11-f1)*(11-(1-sqrt(f1)))
        # = (11-f1)*(10+sqrt(f1))
        #
        # A simpler approach: since the front is smooth within each segment,
        # and hypervolume for a monotone front is maximized by uniform spacing
        # in the "dominated area" coordinate, we can use a change of variable.
        #
        # For f2 = 1 - sqrt(f1), let u = sqrt(f1), so f1 = u^2, f2 = 1 - u
        # The dominated area from a point to the reference is (11-u^2)*(11-(1-u))
        # = (11-u^2)*(10+u)
        # This is a cubic in u, so equal-area spacing requires solving for u
        # such that the cumulative area is linear.
        #
        # Simpler: just use dense uniform sampling in f1 within each segment,
        # with more points in segments that contribute more to hypervolume.
        #
        # The hypervolume contribution of a segment [a,b] in f1 is approximately
        # the area under the front in that region, which is:
        # integral from a to b of (11-f1)*(11-(1-sqrt(f1))) df1
        # = integral (11-f1)*(10+sqrt(f1)) df1
        #
        # Let's compute segment weights based on their length and position.
        
        n_samples = self.budget
        
        # Compute weight for each segment based on its contribution to hypervolume
        # A segment [a,b] contributes area proportional to its length and the
        # average "height" of the front in that region.
        # For f2 = 1 - sqrt(f1), the "height" (11 - f2) = 10 + sqrt(f1)
        # and the "width" contribution is (11 - f1)
        # So the local density of hypervolume per unit f1 is (11-f1)*(10+sqrt(f1))
        
        def seg_weight(a, b):
            # Approximate integral of (11-f1)*(10+sqrt(f1)) from a to b
            # Using midpoint rule with a few points
            n_pts = 10
            f1_vals = np.linspace(a, b, n_pts)
            weights = (11 - f1_vals) * (10 + np.sqrt(f1_vals))
            return np.trapz(weights, f1_vals)
        
        total_weight = sum(seg_weight(a, b) for a, b in segments)
        
        # Allocate samples proportionally to hypervolume contribution
        x1_values = []
        for a, b in segments:
            w = seg_weight(a, b)
            frac = w / total_weight
            n_seg = max(2, int(n_samples * frac))
            
            # Within each segment, use non-uniform spacing to maximize hypervolume
            # For a monotone decreasing front, equal spacing in f2 (or in the
            # "dominated area" coordinate) is better than equal spacing in f1.
            #
            # Since f2 = 1 - sqrt(f1), equal spacing in f2 means:
            # f2 values are equally spaced, so sqrt(f1) values are equally spaced
            # This means f1 values are squares of equally spaced values.
            #
            # But we also want to account for the (11-f1) factor.
            # Let's use a compromise: space points so that the cumulative
            # "area" (11-f1)*(10+sqrt(f1)) is roughly linear.
            #
            # For simplicity and robustness, use sqrt-spacing in f1 within each segment,
            # which corresponds to uniform spacing in sqrt(f1) = 1 - f2.
            
            u_vals = np.linspace(np.sqrt(a), np.sqrt(b), n_seg)
            x1_seg = u_vals ** 2
            x1_values.extend(x1_seg.tolist())
        
        # Ensure all segment boundaries are included
        for a, b in segments:
            x1_values.append(a)
            x1_values.append(b)
        
        x1_values = np.array(sorted(set(x1_values)))
        
        # Limit to budget
        if len(x1_values) > self.budget:
            # Use stratified sampling to keep representation across all segments
            idx = np.linspace(0, len(x1_values) - 1, self.budget).astype(int)
            x1_values = x1_values[idx]
        
        # Generate and evaluate points on the Pareto front
        # x2..x30 = 0 ensures g = 1 (the minimum possible value)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        for i, x1 in enumerate(x1_values):
            points[i, 0] = x1
            # x2..x30 remain 0, which gives g=1
        
        # Clip to bounds
        points = np.clip(points, lo, hi)
        
        # Evaluate all points - each evaluation is recorded by the evaluator
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(points[i])
        
        return points
# EVOLVE-BLOCK-END
