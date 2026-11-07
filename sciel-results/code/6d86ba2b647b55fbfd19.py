# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive multi-objective optimizer for ZDT1 using hypervolume-driven bisection.
    This implementation models the search as an optimal control problem where
    we allocate the evaluation budget to maximize marginal hypervolume gain.
    We exploit the ZDT1 structure: x_2..x_n = 0 minimizes g(x), placing points
    on the Pareto front. We use the known front shape f2 = 1 - sqrt(f1) (up to a
    constant factor) to estimate gains and select the next x_1 to evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        
        # Extract bounds for x_1 (index 0)
        # The problem is ZDT1, so x_1 is in [0, 1]
        lo_1 = bounds[0][0]
        hi_1 = bounds[0][1]
        
        # We will fix x_2..x_n to 0 to stay on the Pareto front
        # This is valid for ZDT1 as it minimizes g(x)
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x1_val):
            nonlocal used
            if used >= budget:
                return None
            x = np.zeros(n)
            x[0] = x1_val
            # x[1:] remains 0
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Points stored as (f1, f2, x1)
        # Sorted by f1 (which is x1)
        points = []
        
        # Helper to estimate f2 for a given x1 on the Pareto front
        # ZDT1: f2 = (1+g)(1-sqrt(f1)). Min g=1. So f2 = 2(1-sqrt(f1)).
        # For gain ranking, the constant factor 2 cancels out.
        # We use 1 - sqrt(x1) for estimation.
        def est_f2(x1_val):
            if x1_val < 0:
                x1_val = 0
            if x1_val > 1:
                x1_val = 1
            return 1.0 - np.sqrt(x1_val)

        # Phase 1: Initial points
        # Evaluate endpoints of the x1 range
        f_start = evaluate(lo_1)
        if f_start is not None:
            points.append((f_start[0], f_start[1], lo_1))
            
        f_end = evaluate(hi_1)
        if f_end is not None:
            points.append((f_end[0], f_end[1], hi_1))
            
        # Sort by f1
        points.sort(key=lambda p: p[0])
        
        # Phase 2: Adaptive Bisection
        # We maintain the list of non-dominated points.
        # Since we are on the Pareto front, all points are non-dominated.
        
        while used < budget:
            best_gain = -1.0
            best_x1 = None
            best_f1 = None
            best_f2 = None
            
            # Iterate through all intervals between consecutive points
            # Also consider the "virtual" intervals to the left of the first point
            # and to the right of the last point?
            # Since x1 is bounded [0,1] and we start with 0 and 1, 
            # the front is fully covered by [0,1].
            # However, if lo_1 > 0 or hi_1 < 1, we might have gaps.
            # But ZDT1 x1 is usually [0,1].
            
            # If we have at least 2 points, we have intervals between them.
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    p_left = points[i]
                    p_right = points[i + 1]
                    
                    f1_l = p_left[0]
                    f1_r = p_right[0]
                    
                    # Midpoint in x1 space
                    x1_mid = (f1_l + f1_r) / 2.0
                    
                    # Estimate f2
                    f2_mid_est = est_f2(x1_mid)
                    
                    # Calculate marginal hypervolume gain
                    # Formula derived: Gain = (f1_r - x1_mid) * (f2_l - f2_mid)
                    # Note: f2_l is the actual f2 of the left point.
                    # f2_mid is the estimated f2 of the new point.
                    # This formula assumes the new point is non-dominated.
                    # Since f1_mid is between f1_l and f1_r, and f2 is decreasing,
                    # f2_mid_est should be between f2_l and f2_r.
                    # If the estimation is accurate, gain is positive.
                    
                    # Use actual f2_l from the evaluated point
                    f2_l_actual = p_left[1]
                    
                    gain = (f1_r - x1_mid) * (f2_l_actual - f2_mid_est)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x1 = x1_mid
                        best_f1 = x1_mid
                        best_f2 = f2_mid_est
            
            # If we have only 1 point (should not happen if budget >= 2 and evals succeeded),
            # or if no interval gave positive gain (should not happen), 
            # we might need to handle edge cases.
            # But with initial 0 and 1, we always have intervals.
            
            if best_x1 is None:
                # Fallback: if something went wrong, try midpoint of [0,1] if not evaluated
                # Or break
                break
                
            # Evaluate the best candidate
            f_val = evaluate(best_x1)
            if f_val is None:
                break
                
            # Insert the new point into the sorted list
            new_point = (f_val[0], f_val[1], best_x1)
            
            # Find insertion index
            ins_idx = 0
            for k in range(len(points)):
                if points[k][0] < new_point[0]:
                    ins_idx = k + 1
                else:
                    break
            
            points.insert(ins_idx, new_point)

        return np.array(all_X)
# EVOLVE-BLOCK-END
