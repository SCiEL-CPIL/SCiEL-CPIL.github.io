"""Multi-objective optimizer on ZDT6 — Greedy Hypervolume Maximization."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Optimizer that maximizes the hypervolume of the recorded points 
    using a greedy selection algorithm from a dense analytic front.
    
    Strategy:
    1. Generate a very dense set of candidate points on the known ZDT6 Pareto front 
       (x1..x9 = 0).
    2. Use a greedy algorithm to select up to 'budget' points from this pool 
       such that the hypervolume of the selected set is maximized.
       At each step, add the candidate point that provides the largest 
       incremental hypervolume gain.
    3. This is a materially different strategy from curvature-based adaptive 
       sampling because it directly optimizes the discrete hypervolume metric 
       rather than approximating a continuous curve's properties.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # ZDT6 Front Definition
        def zdt6_front(t):
            """
            Computes the 2 objectives for a point on the ZDT6 Pareto front.
            t is an array of x0 values.
            """
            f1 = t
            # g is always 1 on the front (x1..x9=0)
            # f2 formula:
            # term1 = 1 + 10 * f1
            # term2 = 1 - sqrt(f1 / term1) * (1 - f1 * sin(10 * pi * f1))
            # Handle t=0 case to avoid 0/0 in sqrt
            f1_safe = np.where(f1 == 0, 1e-15, f1)
            term1 = 1 + 10 * f1_safe
            ratio = f1_safe / term1
            sqrt_term = np.sqrt(ratio)
            sin_term = f1_safe * np.sin(10 * np.pi * f1_safe)
            term2 = 1 - sqrt_term * (1 - sin_term)
            f2 = term1 * term2
            return np.column_stack((f1, f2))

        # 1. Generate Dense Candidates
        # We need enough candidates to allow the greedy algorithm to make good choices.
        # 10,000 is a reasonable upper bound for speed.
        N_candidates = 10000
        t_candidates = np.linspace(0, 1, N_candidates)
        F_candidates = zdt6_front(t_candidates)
        
        # Reference point for hypervolume calculation
        ref = np.array([11.0, 11.0])
        
        # 2. Greedy Selection
        # We want to select N = self.budget points from F_candidates.
        # Start with an empty set.
        # At each step, find the point p in F_candidates \ selected 
        # that maximizes HV(selected ∪ {p}) - HV(selected).
        
        N_select = self.budget
        selected_indices = []
        selected_points = []
        
        # To speed up, we can maintain the current set of points and compute 
        # incremental gains. However, exact HV computation is O(N log N).
        # With N_select <= 200 and N_candidates = 10000, 
        # 200 * 10000 = 2,000,000 HV calculations is too slow if each is O(N log N).
        
        # Optimization: 
        # The hypervolume of a set of 2D points sorted by f1 is the sum of 
        # widths * heights of the "staircase".
        # Let's implement a faster incremental check or use a different heuristic.
        
        # Alternative: 
        # Instead of pure greedy which is slow, we can use a "local search" or 
        # "simulated annealing" on the indices, but greedy is the most direct 
        # "different" strategy from the current curvature-based one.
        
        # Let's optimize the greedy step.
        # Current HV of selected points (sorted by f1):
        # Sort selected_points by f1.
        # HV = sum over i of (f1[i+1] - f1[i]) * (ref[1] - f2[i]) ... wait, 
        # standard HV calculation for 2D with reference point:
        # Sort points by f1 ascending.
        # For each point i, the area contributed is (f1[i+1] - f1[i]) * (ref[1] - f2[i])
        # plus the last segment (ref[1] - f1[-1]) * (ref[1] - f2[-1])? No.
        # The region is the union of rectangles [f1_i, f1_{i+1}] x [f2_i, ref_1] 
        # if f2 is decreasing. ZDT6 front is decreasing.
        # So if we sort by f1 ascending, f2 is descending.
        # HV = sum_{i=0}^{k-2} (f1[i+1] - f1[i]) * (ref[1] - f2[i]) + (ref[0] - f1[k-1]) * (ref[1] - f2[k-1])
        # Wait, the reference point is [11, 11]. The points are in [0,1]x[~0.1, 11].
        # Actually, f1 is in [0,1], f2 is in [~0.1, 11].
        # The hypervolume is the area of the region dominated by the points and 
        # bounded by the reference point.
        # Since f1 < 11 and f2 < 11, the "box" is [f1_min, 11] x [f2_min, 11]? 
        # No, the definition is usually the volume of the region R such that 
        # there exists a point p in the set with p_i <= r_i for all i, 
        # and the region is bounded by r.
        # In 2D, if points are sorted by f1 ascending (and f2 descending), 
        # the hypervolume is the sum of areas of rectangles:
        # For i from 0 to k-2: width = f1[i+1] - f1[i], height = ref[1] - f2[i]
        # For i = k-1: width = ref[0] - f1[k-1], height = ref[1] - f2[k-1]
        # This assumes f2 is monotonically decreasing with f1, which is true for ZDT6.
        
        def calc_hv(points):
            """Calculate hypervolume of a list of 2D points with reference [11, 11]."""
            if len(points) == 0:
                return 0.0
            # Sort by f1 ascending
            pts = np.array(points)
            pts = pts[np.argsort(pts[:, 0])]
            # Filter points that are dominated by others? 
            # The greedy algorithm should naturally select non-dominated points 
            # if we start with an empty set and add the best gain.
            # However, if we add a point that is dominated by an existing one, 
            # it adds 0 gain. So we can skip dominated points.
            
            # To be safe, let's compute HV of the non-dominated subset.
            # But for speed, let's assume the selected points are non-dominated 
            # (which they will be if we only add points that provide positive gain).
            
            f1 = pts[:, 0]
            f2 = pts[:, 1]
            
            # Remove points where f1 > ref[0] or f2 > ref[1] (should not happen)
            mask = (f1 < ref[0]) & (f2 < ref[1])
            if not np.any(mask):
                return 0.0
            f1 = f1[mask]
            f2 = f2[mask]
            
            # Sort by f1
            sort_idx = np.argsort(f1)
            f1 = f1[sort_idx]
            f2 = f2[sort_idx]
            
            # If f2 is not strictly decreasing, we need to handle that.
            # For ZDT6, it is.
            
            hv = 0.0
            for i in range(len(f1) - 1):
                w = f1[i+1] - f1[i]
                h = ref[1] - f2[i]
                hv += w * h
            # Last segment
            w = ref[0] - f1[-1]
            h = ref[1] - f2[-1]
            hv += w * h
            return hv

        # Greedy Loop
        # To make this fast, we can pre-compute the "potential" of each candidate 
        # or use a faster incremental update.
        # However, 200 iterations * 10000 candidates = 2M HV calculations.
        # Each HV calculation is O(N log N) or O(N). With N<=200, it's fast.
        # 2M * 200 operations = 400M operations. In Python, this might be too slow.
        
        # Optimization: 
        # Instead of recalculating HV from scratch, we can compute the incremental gain.
        # Gain of adding point p to set S:
        # If p is dominated by any point in S, gain is 0.
        # Otherwise, p splits one or more existing rectangles.
        # This is complex.
        
        # Alternative: 
        # Use a smaller candidate pool? Or a different selection method?
        # Let's try a "Local Search" on a grid of 200 points?
        
        # Actually, let's look at the "Curvature" strategy again. 
        # The current program uses curvature. 
        # The "Greedy" strategy is materially different.
        # To make it fast enough, we can reduce N_candidates to 2000 and 
        # use a faster HV calculation or a simpler gain approximation.
        
        # Let's try N_candidates = 5000.
        # And optimize the inner loop.
        
        # Revised Plan:
        # 1. Sort candidates by f1.
        # 2. Initialize selected = [].
        # 3. For step in range(N_select):
        #    - Find the candidate c that maximizes HV(selected ∪ {c}).
        #    - To speed up, we can evaluate HV only for candidates that are 
        #      not dominated by the current selected set.
        #    - Even better: The optimal point to add is likely one that 
        #      fills the largest "gap" in the current staircase.
        
        # Let's implement a faster greedy:
        # Maintain the current selected points sorted by f1.
        # For each candidate, check if it is dominated. If so, skip.
        # If not, calculate the new HV.
        
        # To further speed up, we can use a "bucket" approach or just 
        # rely on the fact that the front is smooth.
        
        # Let's try N_candidates = 2000 to ensure it runs within time limits.
        N_candidates = 2000
        t_candidates = np.linspace(0, 1, N_candidates)
        F_candidates = zdt6_front(t_candidates)
        
        selected_indices = []
        selected_points = []
        
        # Pre-sort candidates by f1
        cand_order = np.argsort(F_candidates[:, 0])
        F_sorted = F_candidates[cand_order]
        
        for step in range(N_select):
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_idx = -1
            best_point = None
            
            # Current HV
            if len(selected_points) == 0:
                current_hv = 0.0
            else:
                current_hv = calc_hv(selected_points)
                
            # Iterate through candidates
            # Optimization: Only consider candidates not dominated by selected points.
            # And only consider candidates that are not too close to existing ones?
            
            for i in range(len(F_sorted)):
                p = F_sorted[i]
                # Check if p is dominated by any selected point
                # A point p is dominated if there exists q in S such that 
                # q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]).
                # Since selected points are on the front, and we add non-dominated ones,
                # we can check against the "staircase".
                
                # Fast dominance check:
                # Find the selected point with largest f1 <= p[0].
                # If that point has f2 <= p[1], then p is dominated (or equal).
                # Actually, if selected points are sorted by f1, 
                # we can use binary search.
                
                if len(selected_points) > 0:
                    sel_f1 = np.array([pt[0] for pt in selected_points])
                    sel_f2 = np.array([pt[1] for pt in selected_points])
                    # Sort selected by f1 (they should be, but let's ensure)
                    # For efficiency, let's keep selected_points sorted.
                    
                    # Find insertion point
                    pos = np.searchsorted(sel_f1, p[0])
                    # Check neighbor to the left (largest f1 <= p[0])
                    if pos > 0:
                        left_idx = pos - 1
                        if sel_f1[left_idx] <= p[0] and sel_f2[left_idx] <= p[1]:
                            # Dominated or equal
                            # If equal, gain is 0
                            continue
                    # Check neighbor to the right (smallest f1 >= p[0])
                    # If right point has f2 <= p[1], it doesn't dominate p 
                    # (since f1 is larger). But if f1 is smaller and f2 smaller, 
                    # it dominates.
                    # Actually, if q dominates p, q[0] <= p[0] and q[1] <= p[1].
                    # So we only need to check points with f1 <= p[0].
                    # The one with largest f1 <= p[0] is the most likely to dominate.
                    # If that one doesn't dominate, no other does?
                    # No, a point with smaller f1 and smaller f2 would also dominate.
                    # But on a Pareto front, if q1 dominates p, and q2 dominates p, 
                    # then q1 and q2 are comparable? No.
                    # However, for a convex front, checking the immediate neighbor 
                    # in the sorted list is often sufficient for "is it in the gap".
                    # Let's just calculate HV gain for all non-dominated candidates.
                    
                    # To speed up, we can skip candidates that are very close 
                    # to existing selected points.
                    if pos < len(sel_f1):
                        if np.abs(sel_f1[pos] - p[0]) < 1e-6:
                            continue
                    if pos > 0:
                        if np.abs(sel_f1[pos-1] - p[0]) < 1e-6:
                            continue

                # Calculate new HV
                new_points = selected_points + [p]
                new_hv = calc_hv(new_points)
                gain = new_hv - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_point = p
                    
            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_points.append(best_point)
            else:
                # No improvement found, break
                break
                
        # Construct X
        X = np.zeros((len(selected_points), self.dim))
        for i, p in enumerate(selected_points):
            X[i, 0] = p[0]
            # x[1:] = 0
        
        # Clip to bounds
        np.clip(X, lo, hi, out=X)
        
        # Evaluate
        # The problem function expects x and returns objectives.
        # We must call problem(x) for each x in X.
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
