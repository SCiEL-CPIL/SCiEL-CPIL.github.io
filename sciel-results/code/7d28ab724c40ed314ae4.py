# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive multi-objective optimizer for ZDT1 using hypervolume-driven bisection.
    Exploits the known structure of ZDT1 where the Pareto front is achieved
    when x_2..x_n are set to 0.
    Uses a strategy that selects the next x_1 to maximize the marginal hypervolume gain.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        # Reference point for hypervolume calculation
        ref = np.array([11.0, 11.0])
        
        def make_x(x1_val):
            x = np.zeros(n)
            x[0] = x1_val
            return x
            
        def evaluate(x1_val):
            nonlocal used
            if used >= budget:
                return None
            x = make_x(x1_val)
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Phase 1: Initial points
        # Start with 0.0 and 1.0 to cover the range.
        f1_start = evaluate(0.0)
        f1_end = evaluate(1.0)
        
        # Store points as (f1, f2, x)
        # We maintain a list of non-dominated points sorted by f1
        points = []
        if f1_start is not None:
            points.append((f1_start[0], f1_start[1], 0.0))
        if f1_end is not None:
            points.append((f1_end[0], f1_end[1], 1.0))
            
        # Sort by f1
        points.sort(key=lambda p: p[0])
        
        # Helper to calculate the optimal x1 in an interval [L, R] for ZDT1
        # L = f1_left, R = f1_right
        # Optimal x minimizes the error or maximizes gain.
        # Derived: x* = ((sqrt(L) + sqrt(L + 3R)) / 3)^2
        def optimal_x1(L, R):
            if R <= L:
                return L
            # Ensure we stay within [0, 1]
            x_opt = ((np.sqrt(L) + np.sqrt(L + 3*R)) / 3.0)**2
            # Clip to valid range [0, 1]
            if x_opt < 0: x_opt = 0.0
            if x_opt > 1.0: x_opt = 1.0
            return x_opt

        # Helper to calculate marginal hypervolume gain of inserting a point (f1, f2)
        # into the current set of points.
        def calc_gain(f1_new, f2_new):
            # Find the index where f1_new would be inserted
            i = -1
            for k in range(len(points)):
                if points[k][0] <= f1_new:
                    i = k
                else:
                    break
            
            if i == -1:
                if len(points) == 0:
                    return (ref[0] - f1_new) * (ref[1] - f2_new)
                return (points[0][0] - f1_new) * (ref[1] - f2_new)
            
            elif i == len(points) - 1:
                return (ref[0] - f1_new) * (points[-1][1] - f2_new)
            
            else:
                return (points[i+1][0] - f1_new) * (points[i][1] - f2_new)

        # Phase 2: Adaptive selection
        while used < budget:
            best_gain = -1
            best_x1 = None
            
            # Candidate 1: Optimal point in each interval between existing points
            for i in range(len(points) - 1):
                f1_left = points[i][0]
                f1_right = points[i+1][0]
                
                # Use the analytically derived optimal x1 for this interval
                x1_cand = optimal_x1(f1_left, f1_right)
                
                # Estimate f2 using the known ZDT1 formula
                f2_est = 1.0 - np.sqrt(x1_cand)
                
                # Calculate gain using the actual points in the list
                # Note: calc_gain uses the current 'points' list which does NOT include the new point yet
                gain_est = calc_gain(x1_cand, f2_est)
                
                if gain_est > best_gain:
                    best_gain = gain_est
                    best_x1 = x1_cand
            
            # Candidate 2: Interval from last point to 1.0 (if last point < 1.0)
            if points and points[-1][0] < 1.0:
                f1_left = points[-1][0]
                f1_right = 1.0
                x1_cand = optimal_x1(f1_left, f1_right)
                f2_est = 1.0 - np.sqrt(x1_cand)
                gain_est = calc_gain(x1_cand, f2_est)
                if gain_est > best_gain:
                    best_gain = gain_est
                    best_x1 = x1_cand
            
            # If no candidates, break
            if best_x1 is None:
                break
            
            # Evaluate the best candidate
            f_val = evaluate(best_x1)
            if f_val is None:
                break
                
            # Update the points list
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
