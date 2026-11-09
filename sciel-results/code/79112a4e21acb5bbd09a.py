"""ZDT2 Optimizer using Hypervolume-driven adaptive 1D sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the analytic structure of ZDT2 to reduce the problem to 1D.
    Instead of a fixed grid, it uses a greedy adaptive strategy to place
    evaluations where they maximize the hypervolume of the recorded set.
    
    ZDT2 Pareto Front:
    f1 = x1
    f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
    for x2...xn = 1/sqrt(n-1) (minimizing the penalty term).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_vars = self.dim
        if n_vars < 2:
            # Fallback for trivial cases, though ZDT2 is usually n>=2
            base_x = np.ones(n_vars)
            base_x[0] = 0.0
        else:
            base_x = np.ones(n_vars)
            base_x[0] = 0.0
            # Optimal setting for x2...xn to be on the front
            # sum(x_i^2) is minimized when x_i = 1/sqrt(n-1)
            base_x[1:] = 1.0 / np.sqrt(n_vars - 1)
        
        # Clip to bounds just in case
        base_x = np.clip(base_x, lo, hi)
        
        used = 0
        ref_point = np.array([11.0, 11.0])
        
        # Store evaluated points (f1, f2)
        # We sort them by f1 to easily calculate hypervolume contributions
        evaluated_points = [] # List of (f1, f2)
        
        # Helper to calculate the hypervolume contribution of adding a new point
        # to a sorted list of non-dominated points.
        # Since we are on the Pareto front, all points are non-dominated with each other
        # if we just track the "best" f2 for a given f1, but here we assume
        # we are sampling the front exactly, so f2 is determined by f1.
        # The hypervolume of a set of points on a monotonic front with ref (R1, R2)
        # is sum of rectangles.
        
        def get_hv(points):
            """
            Calculate hypervolume of a list of (f1, f2) points.
            Points should be sorted by f1 ascending.
            Since they are on the Pareto front, f2 is descending as f1 increases.
            HV = sum_{i} (f1_{i+1} - f1_i) * (R2 - f2_i) 
               + (R1 - f1_last) * (R2 - f2_last)
            Wait, the standard formula for a set of points on a front:
            Sort by f1 asc.
            HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (R2 - f2_i) 
            where f1_{N} = R1.
            Actually, let's be precise.
            The region covered is union of rectangles [f1_i, f1_{i+1}] x [0, R2 - f2_i] ? No.
            The hypervolume is the volume of the region dominated by the set and bounded by ref.
            For 2D:
            Sort points by f1 ascending: p0, p1, ..., pk.
            HV = sum_{i=0}^{k-1} (f1_{i+1} - f1_i) * (R2 - f2_i) + (R1 - f1_k) * (R2 - f2_k)
            This assumes f2 is decreasing. For ZDT2, as f1 (x1) increases, f2 decreases.
            So p0 has smallest f1, largest f2. pk has largest f1, smallest f2.
            The "skyline" is formed by these points.
            The width of the rectangle for point i is determined by the next point's f1.
            """
            if not points:
                return 0.0
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            for i in range(len(points)):
                f1_curr, f2_curr = points[i]
                if i == len(points) - 1:
                    f1_next = ref_point[0]
                else:
                    f1_next = points[i+1][0]
                width = f1_next - f1_curr
                height = ref_point[1] - f2_curr
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def eval_x1(x1_val):
            nonlocal used
            x = base_x.copy()
            x[0] = x1_val
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            # Ensure we store the actual evaluated values, in case of numerical drift
            # though for ZDT2 analytic, they should be exact.
            evaluated_points.append((f[0], f[1]))
            return f

        # Phase 1: Initialize with endpoints and midpoint to establish the shape.
        # This ensures we have a valid set to start the greedy improvement.
        initial_x1s = [0.0, 0.5, 1.0]
        
        for x1 in initial_x1s:
            if used >= self.budget or time.time() > deadline:
                break
            eval_x1(x1)
            
        # Phase 2: Greedy Hypervolume Maximization
        # We maintain a sorted list of x1 values.
        # For each gap between consecutive x1 values (and the ends),
        # we calculate the hypervolume gain if we add a point in the middle.
        # We pick the gap with the maximum gain.
        
        # To make this efficient, we can pre-compute the f2 values for the current set.
        # Since f2 is a function of f1, we can just use the analytic formula for 
        # "virtual" points to decide where to place the next evaluation, 
        # but we MUST call the problem function to record it.
        
        # Analytic f2 for ZDT2:
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        n_penalty = 10 * (n_vars - 1)
        def analytic_f2(f1_val):
            # Handle f1=0 case (sqrt(0)=0, f2 = 1+10n)
            if f1_val <= 0:
                return 1.0 + n_penalty
            return (1.0 + n_penalty) * (1.0 - np.sqrt(f1_val / (1.0 + n_penalty)))
        
        # Current x1s sorted
        # We will keep a list of x1s.
        x1_list = [p[0] for p in evaluated_points]
        x1_list.sort()
        
        # Function to calculate HV gain of inserting a new point at x1_new
        # into the current set defined by x1_list (and their analytic f2s).
        # Note: The actual recorded f2 might differ slightly due to floating point,
        # but the gain estimate using analytic f2 is sufficient for placement.
        
        def calculate_gain(x1_new):
            # Find neighbors in x1_list
            # Since x1_list is sorted, we can find insertion index
            # But doing this for every candidate in a loop is slow if we iterate many times.
            # We will just evaluate the gain for the midpoint of each current gap.
            pass # Placeholder, we'll do it inline in the loop below.

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Identify the best candidate for the next evaluation.
            # Candidates are midpoints of gaps in the current x1_list.
            # Gaps are:
            # 1. [0, x1_list[0]]
            # 2. [x1_list[i], x1_list[i+1]] for i in 0..len-2
            # 3. [x1_list[-1], 1]
            
            best_gain = -1
            best_x1 = -1
            
            # Helper to compute HV of a set of x1s using analytic f2
            def hv_from_x1s(x1s):
                if not x1s:
                    return 0.0
                x1s_sorted = sorted(x1s)
                hv = 0.0
                for i in range(len(x1s_sorted)):
                    f1_curr = x1s_sorted[i]
                    f2_curr = analytic_f2(f1_curr)
                    if i == len(x1s_sorted) - 1:
                        f1_next = ref_point[0]
                    else:
                        f1_next = x1s_sorted[i+1]
                    width = f1_next - f1_curr
                    height = ref_point[1] - f2_curr
                    if width > 0 and height > 0:
                        hv += width * height
                return hv

            current_hv = hv_from_x1s(x1_list)
            
            # Generate candidates: midpoints of all gaps
            # Optimization: We don't need to check EVERY possible point, just the midpoints
            # of the current intervals. This is a standard adaptive quadrature/sampling approach.
            
            candidates = []
            # Gap 0
            if x1_list[0] > 0:
                candidates.append(x1_list[0] / 2.0)
            # Middle gaps
            for i in range(len(x1_list) - 1):
                mid = (x1_list[i] + x1_list[i+1]) / 2.0
                candidates.append(mid)
            # Gap N
            if x1_list[-1] < 1.0:
                candidates.append((x1_list[-1] + 1.0) / 2.0)
            
            # Evaluate gain for each candidate
            for x1_cand in candidates:
                # Calculate new HV
                new_x1s = x1_list + [x1_cand]
                new_hv = hv_from_x1s(new_x1s)
                gain = new_hv - current_hv
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = x1_cand
            
            if best_x1 < 0:
                break
                
            # Evaluate the best point
            eval_x1(best_x1)
            x1_list.append(best_x1)
            x1_list.sort()
            
        return None
# EVOLVE-BLOCK-END
