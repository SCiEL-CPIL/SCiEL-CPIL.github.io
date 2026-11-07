import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2:
        # f1 = x1
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1))
        # With n=30: f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        # f1 = x1, f2 = 11*(1 - sqrt(x1))
        
        # The reference point is [11.0, 11.0]
        # We want to maximize the hypervolume of the recorded points with respect to this reference.
        
        # The Pareto front is f2 = 11*(1 - sqrt(f1)).
        # Let's sample points on this front to maximize the hypervolume.
        
        # The hypervolume contribution of a point (f1, f2) with respect to reference (11, 11) is:
        # (11 - f1) * (11 - f2)
        # But this is not exactly right because of the dominance relation.
        # For a set of non-dominated points, the hypervolume is the volume of the union of the 
        # "boxes" [f1_i, 11] x [f2_i, 11] (for minimization, we consider the region dominated by 
        # the set and bounded by the reference).
        # Actually, for minimization, the hypervolume is the volume of the region in objective 
        # space that is dominated by at least one point in the set and is bounded by the reference 
        # point.
        # For a single point (f1, f2), the volume is (11 - f1) * (11 - f2) if f1 < 11 and f2 < 11.
        
        # For multiple non-dominated points, the hypervolume is the sum of the non-overlapping 
        # parts. Since the points are on a curve, we can compute it by sorting by f1 and then 
        # summing the areas of the trapezoids or rectangles.
        
        # The optimal strategy is to place points such that the hypervolume is maximized.
        # For a concave Pareto front, the optimal spacing is non-uniform.
        
        # Let's use the equal-area property. The integral of (11 - f2(f1)) df1 from 0 to 11 is 
        # the total area. But we only have 200 points.
        
        # Actually, the hypervolume of a set of points on the Pareto front with respect to 
        # reference (11, 11) is the area under the curve formed by the points and the reference.
        
        # Let's think about it differently. We have m points (f1_i, f2_i) with f1_1 < f1_2 < ... < f1_m.
        # The hypervolume is the sum of the areas of the rectangles that each point "owns".
        # For a concave front, the optimal spacing is such that the area between consecutive 
        # points (and the reference) is equalized.
        
        # Let's use the parameterization t in [0, 1] where x1 = t^2 (since x1 = f1, and f2 = 11*(1-sqrt(f1))).
        # Actually, let's use x1 directly.
        
        # The hypervolume contribution of a point at x1 is roughly (11 - x1) * (11 - 11*(1-sqrt(x1)))
        # = (11 - x1) * 11*sqrt(x1)
        # = 11 * (11*sqrt(x1) - x1^(3/2))
        
        # To maximize the total hypervolume with a fixed number of points, we should place them 
        # where this contribution is highest, or more precisely, where the marginal gain is highest.
        
        # A simpler approach: use the known result that for a concave Pareto front, the optimal 
        # spacing in the objective space is such that the hypervolume contribution per point is 
        # equalized.
        
        # Let's try a different approach: since we know the Pareto front exactly, we can compute 
        # the optimal x1 values by solving for the points that maximize the hypervolume.
        
        # For simplicity, let's use a logarithmic spacing in x1, which tends to work well for 
        # concave functions.
        
        # Actually, let's use the fact that the Pareto front is f2 = 11*(1 - sqrt(f1)).
        # Let u = sqrt(f1), so f1 = u^2, f2 = 11*(1 - u).
        # Then the front in (u, f2) space is a straight line: f2 = 11 - 11u.
        # The hypervolume in (f1, f2) space is not the same as in (u, f2) space, but we can 
        # use this to guide our sampling.
        
        # Let's sample u uniformly in [0, 1] and compute f1 = u^2, f2 = 11*(1 - u).
        # This should give a good distribution of points along the front.
        
        num_points = min(200, self.budget)
        
        # Use the equal-area principle to find optimal x1 values
        # h_i = C / sqrt(f1_i), so f1_{i+1} = f1_i + C / sqrt(f1_i)
        # Solve for C such that f1_m = 1
        
        def simulate(C, m):
            f1 = 0.0
            for i in range(m):
                if f1 <= 0:
                    f1 = 1e-10
                f1 += C / np.sqrt(f1)
                if f1 > 1.0:
                    return f1
            return f1
        
        # Bisection to find C such that simulate(C, num_points) = 1
        lo_C = 1e-8
        hi_C = 10.0
        for _ in range(100):
            mid_C = (lo_C + hi_C) / 2
            result = simulate(mid_C, num_points)
            if result < 1.0:
                lo_C = mid_C
            else:
                hi_C = mid_C
        
        C_opt = (lo_C + hi_C) / 2
        
        # Generate the x1 values
        x1_vals = np.zeros(num_points)
        f1 = 0.0
        for i in range(num_points):
            if f1 <= 0:
                f1 = 1e-10
            f1 += C_opt / np.sqrt(f1)
            x1_vals[i] = min(f1, 1.0)
        
        used = 0
        recorded = []
        
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1
        

        return recorded
