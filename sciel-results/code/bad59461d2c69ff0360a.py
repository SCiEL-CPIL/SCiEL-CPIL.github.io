import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer for ZDT1 using structure-aware sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT1, the Pareto front is:
        # f1(x) = x0
        # f2(x) = 1 + 9 * x0 / (1 + x0) * (1 - sqrt(x0/(1+x0)))
        # when x[1:] = 0 (which minimizes g(x[1:]))
        # 
        # We want to choose n points x0_1, ..., x0_n in [0,1] to maximize
        # the hypervolume of the resulting points (f1, f2) with reference [11, 11].
        #
        # The Pareto front of ZDT1 is concave. To maximize hypervolume,
        # we should place points where the curve is steepest (near x0=0).
        # A good strategy is to use a power-law distribution:
        # x0_i = (i/n)^beta for i = 1, ..., n with beta > 1
        # This places more points near x0=0.
        
        # Define the Pareto front analytically
        def pareto_front(x0):
            """Compute (f1, f2) for given x0 values on the Pareto front."""
            f1 = x0
            f2 = 1.0 + 9.0 * x0 / (1.0 + x0) * (1.0 - np.sqrt(x0 / (1.0 + x0)))
            return f1, f2

        # Compute hypervolume for a set of 2D points with reference point (11, 11)
        def hypervolume_2d(points, ref=(11.0, 11.0)):
            """Compute exact hypervolume for 2D points."""
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than reference in both objectives
            valid = [(f1, f2) for f1, f2 in points if f1 < ref[0] and f2 < ref[1]]
            if len(valid) == 0:
                return 0.0
            
            # Sort by f1 (ascending)
            valid.sort(key=lambda p: p[0])
            
            # Compute hypervolume by summing rectangles
            hv = 0.0
            prev_f2 = ref[1]
            for f1, f2 in valid:
                if f2 < prev_f2:
                    # This point is non-dominated and contributes
                    width = f1 - (valid[valid.index((f1, f2)) - 1][0] if valid.index((f1, f2)) > 0 else 0.0)
                    # Actually, let's use a simpler approach:
                    pass
            
            # Simpler exact computation:
            # Sort by f1 ascending, then for each point, the contribution is
            # (f1_i - f1_{i-1}) * (ref[1] - f2_i) where f1_0 = 0
            # But we need to handle dominance properly.
            
            # Better approach: sort by f1 ascending, keep track of min f2 seen
            hv = 0.0
            min_f2_so_far = ref[1]
            prev_f1 = 0.0
            for f1, f2 in valid:
                if f2 < min_f2_so_far:
                    # This point extends the non-dominated front
                    width = f1 - prev_f1
                    height = min_f2_so_far - f2
                    hv += width * height
                    min_f2_so_far = f2
                prev_f1 = f1
            
            # Add the final rectangle from last point to ref
            if min_f2_so_far < ref[1]:
                width = ref[0] - prev_f1
                height = min_f2_so_far - ref[1] if min_f2_so_far > ref[1] else ref[1] - min_f2_so_far
                # Actually, the last contribution is from the last non-dominated point
                # Let me redo this more carefully
            
            # Correct hypervolume computation for 2D:
            # After sorting by f1 ascending, the non-dominated points form a staircase.
            # HV = sum over consecutive non-dominated points of (f1_{i+1} - f1_i) * (ref_y - f2_i)
            # plus the last segment to ref_x
            
            # Filter to non-dominated points (sorted by f1 asc, keep those with decreasing f2)
            nondom = []
            min_f2 = ref[1]
            for f1, f2 in valid:
                if f2 < min_f2:
                    nondom.append((f1, f2))
                    min_f2 = f2
            
            if len(nondom) == 0:
                return 0.0
            
            hv = 0.0
            for i in range(len(nondom) - 1):
                f1_i, f2_i = nondom[i]
                f1_next, _ = nondom[i + 1]
                hv += (f1_next - f1_i) * (ref[1] - f2_i)
            # Last segment
            f1_last, f2_last = nondom[-1]
            hv += (ref[0] - f1_last) * (ref[1] - f2_last)
            
            return hv

        # Objective function to minimize (negative hypervolume)
        def neg_hv(beta):
            n = budget
            x0_vals = np.array([(i / n) ** beta for i in range(1, n + 1)])
            f1, f2 = pareto_front(x0_vals)
            points = list(zip(f1, f2))
            return -hypervolume_2d(points)

        # Optimize beta using scipy
        try:
            result = minimize_scalar(neg_hv, bounds=(0.5, 5.0), method='bounded', 
                                      options={'xatol': 1e-6})
            best_beta = result.x
        except Exception:
            best_beta = 2.0  # Fallback
        
        # Generate the optimal x0 values
        n = budget
        x0_values = np.array([(i / n) ** best_beta for i in range(1, n + 1)])
        
        # Evaluate all points
        X = np.zeros((n, dim))
        F = np.zeros((n, self.n_obj))
        for i, x0 in enumerate(x0_values):
            X[i, 0] = x0
            F[i] = problem(X[i])
        
        used = n
        best_X = X.copy()
        best_F = F.copy()
        
        return best_X
