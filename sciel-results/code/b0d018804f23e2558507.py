# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
            
        # 1. Initialization using Latin Hypercube Sampling (LHS)
        # Divide each dimension into pop_size equal intervals and sample one point from each
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            intervals = np.linspace(lo[d], hi[d], pop_size + 1)
            for i in range(pop_size):
                # Random point within interval i
                X[i, d] = rng.uniform(intervals[i], intervals[i+1])
                
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper: Calculate hypervolume for 3 objectives
        def calc_hv(F_set, ref):
            # Filter points strictly better than ref
            mask = np.all(F_set < ref, axis=1)
            F_set = F_set[mask]
            if len(F_set) == 0:
                return 0.0
            
            # For 3D, we can sort by first objective and sweep
            # Or use a simple decomposition
            # Given small N, a robust O(N^2) or O(N^3) method is fine
            # Let's use a simple inclusion-exclusion or grid-like approach?
            # Actually, for N=20, we can just compute the volume of the union of boxes.
            # Box for point i: [F_i[0], ref[0]] x [F_i[1], ref[1]] x [F_i[2], ref[2]]
            # This is complex. Let's use a standard 3D HV algorithm.
            
            # Sort by first objective (ascending)
            idx = np.argsort(F_set[:, 0])
            F_sorted = F_set[idx]
            
            hv = 0.0
            # We can decompose into "shells"
            # For each point in sorted order, the unique volume it contributes is
            # (ref[0] - F_prev[0]) * HV_2D(points with f1 > F_prev[0] and f1 <= F_curr[0]...)
            # This is getting complicated to implement correctly from scratch quickly.
            
            # Alternative: Use a simple approximation or a known library?
            # No external libs for HV.
            # Let's use a simple O(N^3) method:
            # The HV is the volume of the union of hyper-rectangles.
            # For 3D, we can project onto 2D for each "layer" in the first dimension.
            
            # Better: Just use the non-dominated property.
            # If we maintain a non-dominated set, the HV is the sum of contributions.
            # Contribution of point i is the volume of the box [F_i, ref] that is not covered by any other point j where F_j <= F_i.
            # Since all are non-dominated, no F_j <= F_i for all j.
            # But F_j can be better in some dims and worse in others.
            
            # Let's implement a standard 3D HV calculation:
            # 1. Filter non-dominated points (should already be done, but just in case)
            # 2. Sort by x1
            # 3. For each point i (in sorted order), calculate the area of the 2D region in x2-x3 plane
            #    that is not dominated by any previous point (with smaller x1) and is within [x2_i, ref2] x [x3_i, ref3].
            #    The volume contribution is (x1_{i-1} - x1_i) * Area. Wait, x1 is increasing.
            #    The "width" in x1 for point i is (x1_{i-1} - x1_i) ? No.
            #    Let's sort ascending by x1.
            #    The region dominated by point i but not by any j with x1_j < x1_i is a box of width (x1_{prev} - x1_i) ?
            #    Actually, the unique contribution of point i in a sorted list (ascending x1) is:
            #    Width in x1: (x1_{i-1} - x1_i) is wrong direction.
            #    Let's define the "slice" for point i as [x1_i, x1_{i-1}] where i-1 is the previous point in sorted order?
            #    No, the standard algorithm:
            #    Sort by x1 ascending.
            #    For i from N-1 down to 0:
            #       The point i dominates a region in x2,x3 that is not dominated by points j > i (which have larger x1, so they don't dominate in x1).
            #       Wait, if x1_j > x1_i, j cannot dominate i in x1.
            #       So the unique volume of i is the volume of the box [x1_i, x1_{i+1}] (or ref) x [x2_i, ref2] x [x3_i, ref3]
            #       MINUS the parts covered by points k > i that have x2_k < x2_i and x3_k < x3_i?
            #       This is essentially calculating the 2D HV of the "shadow" cast by points with larger x1.
            
            # Let's use a simpler, slower but correct O(N^3) method for N=20.
            # HV = Sum over all pairs? No.
            # Let's just use a Monte Carlo? No, need exact.
            
            # Correct 3D HV Algorithm (O(N^2) after sorting):
            # 1. Sort by x1 ascending.
            # 2. Initialize a list of "active" points for 2D HV.
            # 3. Iterate from largest x1 to smallest.
            # 4. For current point p, its contribution is (x1_next - x1_curr) * HV_2D(active_points, p.x2, p.x3, ref2, ref3)
            #    where active_points are points already processed (larger x1) that have x2 < p.x2 and x3 < p.x3?
            #    No, the 2D HV is calculated on the points that are "behind" p in x1.
            #    Actually, the volume contributed by p is the volume of the box [x1_p, x1_next] x [x2_p, ref2] x [x3_p, ref3]
            #    that is NOT covered by any point q with x1_q > x1_p.
            #    A point q covers part of p's box if x2_q < x2_p and x3_q < x3_p? No, if x2_q < x2_p, it covers [x2_q, x2_p] in x2?
            #    Yes.
            
            # Let's implement HV_2D first.
            def hv_2d(points, ref2, ref3):
                # points: list of (x2, x3)
                # ref: (ref2, ref3)
                # Sort by x2 ascending
                if len(points) == 0: return 0.0
                pts = sorted(points, key=lambda p: p[0])
                hv = 0.0
                max_x3 = 0.0
                prev_x2 = 0.0
                # Iterate from right (largest x2) to left
                # Actually, standard 2D HV:
                # Sort by x2 descending.
                # Keep track of max x3 seen so far.
                # Contribution of point i: (x2_i - x2_{i+1}) * (ref3 - max_x3_so_far) ?
                # Let's do:
                pts_desc = sorted(points, key=lambda p: -p[0])
                current_max_x3 = 0.0
                prev_x2 = ref2
                for x2, x3 in pts_desc:
                    if x2 >= ref2: continue
                    if x3 >= ref3: continue
                    if x3 > current_max_x3:
                        width = prev_x2 - x2
                        height = ref3 - x3
                        # The area added is width * (ref3 - x3) ?
                        # No, the area is the rectangle [x2, prev_x2] x [x3, ref3]
                        # But parts might be covered by previous points?
                        # Since we iterate x2 descending, and keep max_x3, the "uncovered" height is ref3 - max(max_x3, x3)?
                        # Actually, if x3 > current_max_x3, then the strip [x2, prev_x2] x [x3, ref3] is new.
                        # And the strip [x2, prev_x2] x [current_max_x3, x3] is also new? No, that was covered?
                        # Wait.
                        # Standard algorithm:
                        # Sort x2 descending.
                        # max_x3 = 0
                        # for p in sorted:
                        #   if p.x3 > max_x3:
                        #     hv += (prev_x2 - p.x2) * (ref3 - max_x3)  # No, this is wrong.
                        
                        # Correct 2D HV:
                        # Sort by x2 ascending.
                        # max_x3 = 0
                        # prev_x2 = 0
                        # for p in sorted:
                        #   if p.x3 > max_x3:
                        #     hv += (p.x2 - prev_x2) * (ref3 - max_x3)
                        #     max_x3 = p.x3
                        #     prev_x2 = p.x2
                        #   else:
                        #     # p is dominated in x3 by a previous point with smaller x2?
                        #     # No, if x3 <= max_x3, it doesn't add new volume in the "upper right" sense?
                        #     # Actually, if x3 <= max_x3, the point is dominated by some point with smaller x2 and larger x3.
                        #     # So it contributes 0?
                        #     # Yes, in the context of "skyline".
                        # After loop:
                        # hv += (ref2 - prev_x2) * (ref3 - max_x3)
                        
                        pass # Placeholder, will implement properly below

            # Let's just implement the 2D HV properly inside the 3D function.
            
            # 3D HV Implementation
            # Filter non-dominated
            nd_pts = []
            for i in range(len(F_set)):
                dominated = False
                for j in range(len(F_set)):
                    if i == j: continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_pts.append(F_set[i])
            
            if not nd_pts: return 0.0
            
            # Sort by x1 ascending
            nd_pts.sort(key=lambda p: p[0])
            
            hv = 0.0
            # We process from largest x1 to smallest.
            # Maintain a list of "active" points in 2D (x2, x3) that have larger x1.
            # For each point p, the volume it contributes is (x1_next - x1_p) * HV_2D(active_pts, p.x2, p.x3)
            # Where HV_2D is the hypervolume of the active points in the 2D plane, bounded by p.x2, p.x3 and ref2, ref3.
            # Wait, the active points have LARGER x1. They dominate in x1.
            # So they cover the region [x1_p, x1_next] x [x2_active, ref2] x [x3_active, ref3].
            # The unique volume of p is the part of [x1_p, x1_next] x [x2_p, ref2] x [x3_p, ref3] NOT covered by active points.
            # This is equivalent to:
            # Width * ( Area([x2_p, ref2] x [x3_p, ref3]) - Area(covered by active points in this rectangle) )
            
            active_2d = [] # List of (x2, x3)
            for i in range(len(nd_pts) - 1, -1, -1):
                p = nd_pts[i]
                x1_next = nd_pts[i+1][0] if i < len(nd_pts) - 1 else ref_point[0]
                width = x1_next - p[0]
                
                # Calculate 2D HV of active_2d points, bounded by [p[1], ref2] and [p[2], ref3]
                # Filter active points that are within this bound?
                # No, we need the area of the union of boxes [x2_a, ref2] x [x3_a, ref3] for a in active_2d,
                # INTERSECTED with [p[1], ref2] x [p[2], ref3].
                # This is the same as calculating HV_2D of active_2d with reference (ref2, ref3),
                # but only counting area in the rectangle [p[1], ref2] x [p[2], ref3].
                
                # Let's write a helper: calc_hv_2d_bounded(points, bound_x2, bound_x3, ref2, ref3)
                # Points are (x2, x3).
                # We want the area of union of [x2, ref2]x[x3, ref3] for x in points,
                # clipped to [bound_x2, ref2] x [bound_x3, ref3].
                
                def calc_hv_2d_bounded(pts, bx2, bx3, r2, r3):
                    # Filter points that are within bounds?
                    # A point (x2, x3) contributes if x2 < r2 and x3 < r3.
                    # Its box is [x2, r2] x [x3, r3].
                    # We want the area of the union of these boxes, intersected with [bx2, r2] x [bx3, r3].
                    # This is equivalent to the area of the union of boxes [max(x2, bx2), r2] x [max(x3, bx3), r3].
                    # So we can just adjust the points: x2' = max(x2, bx2), x3' = max(x3, bx3).
                    # Then calculate standard 2D HV of these adjusted points with ref (r2, r3).
                    
                    adjusted = []
                    for x2, x3 in pts:
                        ax2 = max(x2, bx2)
                        ax3 = max(x3, bx3)
                        if ax2 < r2 and ax3 < r3:
                            adjusted.append((ax2, ax3))
                    
                    if not adjusted: return 0.0
                    
                    # Standard 2D HV
                    adjusted.sort(key=lambda p: p[0])
                    hv = 0.0
                    max_x3 = 0.0
                    prev_x2 = 0.0
                    for x2, x3 in adjusted:
                        if x3 > max_x3:
                            hv += (x2 - prev_x2) * (r3 - max_x3)
                            max_x3 = x3
                            prev_x2 = x2
                    hv += (r2 - prev_x2) * (r3 - max_x3)
                    return hv

                covered_area = calc_hv_2d_bounded(active_2d, p[1], p[2], ref_point[1], ref_point[2])
                total_area = (ref_point[1] - p[1]) * (ref_point[2] - p[2])
                unique_area = total_area - covered_area
                
                hv += width * unique_area
                
                # Add current point to active list for next iteration (smaller x1)
                active_2d.append((p[1], p[2]))
                
            return hv

        # Main Loop
        # We will use a population of 20.
        # We will keep track of all evaluated points to calculate final HV?
        # No, the task says "Every returned objective vector is recorded".
        # So we just need to call `problem` and let the evaluator record it.
        # We don't need to store them all, just manage the population for search.
        
        # To select parents and decide replacements, we maintain X and F for the population.
        
        # Adaptive step size
        step_size = 0.5 * np.max(range_vec)
        
        # We will run until evals < 150
        while evals < 150:
            if time.time() > deadline: break
            
            # Identify non-dominated in current population
            non_dom_idx = []
            for i in range(pop_size):
                is_dom = False
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            
            if len(non_dom_idx) == 0:
                # Fallback
                parent_idx = np.argmin(np.sum(F, axis=1))
            else:
                # Pick random non-dominated
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                
            parent = X[parent_idx]
            
            # Mutation
            mutation = rng.standard_normal(self.dim) * step_size
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Update population
            # Add child to a temporary list, then select best 20
            # To be efficient, we just replace the worst point.
            # Determine worst point in current population + child.
            
            # Create a combined list
            X_combined = np.vstack([X, child.reshape(1, -1)])
            F_combined = np.vstack([F, fc.reshape(1, -1)])
            
            # We want to keep 20 points.
            # Remove the one with lowest HV contribution.
            # This is O(N^2) or more. N=21.
            # For each point i, calc HV(S) - HV(S\i).
            # HV(S) is constant. So just calc HV(S\i).
            # This is 21 * HV_calc. HV_calc is O(N^2) or so. 21*400 = 8400 ops. Fine.
            
            base_hv = calc_hv(F_combined, ref_point)
            
            min_contrib = float('inf')
            worst_idx = 0
            for i in range(len(F_combined)):
                if i >= len(F_combined): break
                # Create set without i
                mask = np.ones(len(F_combined), dtype=bool)
                mask[i] = False
                F_reduced = F_combined[mask]
                hv_reduced = calc_hv(F_reduced, ref_point)
                contrib = base_hv - hv_reduced
                if contrib < min_contrib:
                    min_contrib = contrib
                    worst_idx = i
            
            # Remove worst_idx
            X = np.delete(X_combined, worst_idx, axis=0)
            F = np.delete(F_combined, worst_idx, axis=0)
            
            # Adjust step size
            # Check if child is non-dominated in new population
            child_is_nd = False
            child_pos = np.argmin(np.linalg.norm(X - child, axis=1)) # Find where child is
            is_dom = False
            for j in range(pop_size):
                if j == child_pos: continue
                if np.all(F[j] <= F[child_pos]) and np.any(F[j] < F[child_pos]):
                    is_dom = True
                    break
            if not is_dom:
                child_is_nd = True
                
            if child_is_nd:
                step_size = min(1.0, step_size * 1.05)
            else:
                step_size = max(0.05, step_size * 0.95)

        # Phase 2: Local Polish
        # Use remaining budget
        if evals < self.budget:
            # Get non-dominated points
            non_dom_idx = []
            for i in range(pop_size):
                is_dom = False
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            
            # Sort by sum of objectives (best first)
            non_dom_idx.sort(key=lambda i: np.sum(F[i]))
            
            # Polish top 5
            for idx in non_dom_idx[:5]:
                if evals >= self.budget: break
                if time.time() > deadline: break
                
                x0 = X[idx]
                
                def scalarized(x):
                    nonlocal evals
                    if evals >= self.budget:
                        return 1e10
                    if time.time() > deadline:
                        return 1e10
                    f = problem(x)
                    evals += 1
                    return np.sum(f)
                
                try:
                    res = minimize(
                        scalarized, x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'maxfun': 10}
                    )
                    # The minimize function calls scalarized, which increments evals.
                    # We need to check if the result is better.
                    # The result.x is the polished point.
                    # We need to evaluate it once more to get the vector?
                    # No, scalarized returned the sum, but we didn't store the vector.
                    # We should store the best vector found during minimization?
                    # Or just evaluate res.x again.
                    if evals < self.budget and time.time() < deadline:
                        new_f = problem(res.x)
                        evals += 1
                        # Check if new_f is better than F[idx]
                        # Simple check: if new_f dominates F[idx] or has better sum
                        if np.all(new_f <= F[idx]) or np.sum(new_f) < np.sum(F[idx]):
                            X[idx] = res.x
                            F[idx] = new_f
                except:
                    pass

        return X

# EVOLVE-BLOCK-END
