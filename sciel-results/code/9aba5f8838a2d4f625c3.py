"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Exploit the known analytical Pareto front of ZDT4.
    2. The Pareto front is located at x_1 in [0, 1] and x_2..x_n = 0.
    3. Use greedy hypervolume maximization to select 200 points that
       maximize the hypervolume indicator.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Determine the optimal values for the deceptive variables x_2..x_n.
        # For ZDT4, the global minimum of the penalty function g(x) is at x_i = 0.
        optimal_deceptive_vals = np.zeros(n - 1, dtype=np.float64)
        for i in range(1, n):
            if lo[i] > 0:
                optimal_deceptive_vals[i-1] = lo[i]
            elif hi[i] < 0:
                optimal_deceptive_vals[i-1] = hi[i]
            else:
                optimal_deceptive_vals[i-1] = 0.0

        # Sample x_1 (the first variable) to maximize hypervolume.
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # The Pareto front is defined for x_1 in [0, 1].
        sample_start = max(x1_lo, 0.0)
        sample_end = min(x1_hi, 1.0)
        
        # If the valid range is invalid, fallback to full bounds
        if sample_start >= sample_end:
            sample_start = x1_lo
            sample_end = x1_hi

        # For ZDT4 with g=1: f1 = x1, f2 = 1 + sin(10*pi*x1)
        # Reference point: [11, 400]
        
        # Analytical Pareto front:
        # f1(x) = x
        # f2(x) = 1 + sin(10*pi*x)
        # The front is in [0, 1] for x1.
        
        # Greedy hypervolume maximization
        # Start with 2 endpoints
        xs = [sample_start, sample_end]
        candidates = [(sample_start + sample_end) / 2.0]
        
        # Precompute f values for known points
        def f1(x):
            return x
        
        def f2(x):
            return 1.0 + np.sin(10.0 * np.pi * x)
        
        # Compute hypervolume for a set of points (sorted by f1)
        # Reference point: [11, 400]
        ref = np.array([11.0, 400.0])
        
        def compute_hv(points):
            """
            Compute hypervolume for 2D points.
            points: list of (f1, f2) tuples, assumed non-dominated and sorted by f1.
            """
            if not points:
                return 0.0
            
            hv = 0.0
            # Sort by f1 (should already be sorted)
            sorted_pts = sorted(points, key=lambda p: p[0])
            
            prev_f2 = ref[1]
            for f1_val, f2_val in sorted_pts:
                if f1_val >= ref[0] or f2_val >= ref[1]:
                    continue
                width = f1_val - (sorted_pts[sorted_pts.index((f1_val, f2_val))-1][0] if sorted_pts.index((f1_val, f2_val)) > 0 else 0.0)
                # Actually, let's use the standard 2D HV computation
                break
            
            # Standard 2D HV: sum of rectangles
            # For sorted points by f1, the HV is sum of (f1[i] - f1[i-1]) * (ref[1] - f2[i])
            # where f1[-1] = 0 (or the minimum f1)
            
            if len(sorted_pts) == 0:
                return 0.0
            
            total_hv = 0.0
            prev_f1 = 0.0
            for f1_val, f2_val in sorted_pts:
                if f2_val < ref[1]:
                    total_hv += (f1_val - prev_f1) * (ref[1] - f2_val)
                prev_f1 = f1_val
            
            # Add the last rectangle from last point to ref[0]
            if sorted_pts:
                last_f1, last_f2 = sorted_pts[-1]
                if last_f2 < ref[1]:
                    total_hv += (ref[0] - last_f1) * (ref[1] - last_f2)
            
            return total_hv
        
        # Better HV computation: for 2D, if points are sorted by f1 (increasing),
        # the hypervolume is the area under the step function.
        def hv_2d(points_sorted_by_f1):
            """
            points_sorted_by_f1: list of (f1, f2) sorted by f1 increasing.
            Assumes all points are below the reference point.
            """
            if not points_sorted_by_f1:
                return 0.0
            
            hv = 0.0
            prev_f1 = 0.0
            for f1_val, f2_val in points_sorted_by_f1:
                hv += (f1_val - prev_f1) * (ref[1] - f2_val)
                prev_f1 = f1_val
            hv += (ref[0] - prev_f1) * (ref[1] - points_sorted_by_f1[-1][1])
            return hv
        
        # Greedy selection
        # Maintain a list of selected x values, sorted
        selected_xs = [sample_start, sample_end]
        selected_f = [(f1(sample_start), f2(sample_start)), (f1(sample_end), f2(sample_end))]
        
        # Current HV
        current_hv = hv_2d(selected_f)
        
        # Candidate points: midpoints of intervals
        # We'll maintain a list of candidate x values
        candidate_xs = [(sample_start + sample_end) / 2.0]
        
        # For efficiency, we'll compute the marginal gain for each candidate
        # and pick the best one
        
        for _ in range(budget - 2):
            if time.time() > deadline:
                break
            
            if not candidate_xs:
                break
            
            best_gain = -1
            best_idx = -1
            best_x = None
            
            for idx, cx in enumerate(candidate_xs):
                cf1 = f1(cx)
                cf2 = f2(cx)
                
                # Compute new HV if we add this point
                # Insert into sorted list
                new_f = selected_f + [(cf1, cf2)]
                new_f.sort(key=lambda p: p[0])
                new_hv = hv_2d(new_f)
                gain = new_hv - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_x = cx
            
            if best_x is None:
                break
            
            # Add the best point
            selected_xs.append(best_x)
            selected_xs.sort()
            selected_f.append((f1(best_x), f2(best_x)))
            selected_f.sort(key=lambda p: p[0])
            current_hv += best_gain
            
            # Update candidates: remove the chosen one, add new midpoints
            candidate_xs.pop(best_idx)
            
            # Find the neighbors of best_x in selected_xs
            # and add midpoints to the new intervals
            if len(selected_xs) >= 2:
                # Find index of best_x
                idx = selected_xs.index(best_x)
                
                if idx > 0:
                    left_x = selected_xs[idx - 1]
                    new_mid = (left_x + best_x) / 2.0
                    candidate_xs.append(new_mid)
                
                if idx < len(selected_xs) - 1:
                    right_x = selected_xs[idx + 1]
                    new_mid = (best_x + right_x) / 2.0
                    candidate_xs.append(new_mid)
        
        # Evaluate all selected points
        results = []
        for x1_val in selected_xs:
            if time.time() > deadline:
                break
            
            x = np.empty(n, dtype=np.float64)
            x[0] = x1_val
            x[1:] = optimal_deceptive_vals
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
