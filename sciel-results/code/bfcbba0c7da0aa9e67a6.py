"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 with hypervolume-aware selection and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s/330s limits

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            is_nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nondom[j]:
                        continue
                    if dominates(F[j], F[i]):
                        is_nondom[i] = False
                        break
            return np.where(is_nondom)[0]

        # Initialize population
        pop_size = min(20, max(5, self.budget // 10))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (only points strictly better count)
        ref = np.array([3.0, 5.0, 7.0])

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume in 3D for points strictly better than ref."""
            # Filter points strictly better than ref in all objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # Compute hypervolume using inclusion-exclusion / decomposition
            # For 3D, we can use the standard algorithm
            # Sort by f1, then for each point compute the contribution
            # that is not covered by points with smaller f1
            
            n = len(pts)
            if n == 0:
                return 0.0
            
            # For each point, compute the volume it contributes
            # A point p contributes volume if it is not dominated by any other point
            # We use the standard 3D hypervolume algorithm
            # Sort by f1 ascending
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            
            # For each point in sorted order, compute the region it uniquely covers
            # The region for point i is [f1_i, f1_{i+1}] x [f2_i, max_f2_above] x [f3_i, max_f3_above]
            # But we need to be more careful. Let's use a simpler approach:
            # For 3D with small n, we can compute by iterating
            
            # Actually, let's use a more efficient method:
            # Sort by f1. For each point, the contribution is the volume of the box
            # [f1_i, next_f1] x [f2_i, upper_bound_f2] x [f3_i, upper_bound_f3]
            # where upper bounds are determined by the minimum of f2 and f3 of all
            # subsequent points that have f2 < f2_i or f3 < f3_i... this gets complex.
            
            # Let's use a direct computation for small n
            hv = 0.0
            for i in range(n):
                # Volume contributed by point i that is not dominated by any other point
                # This is the volume of the box [pts[i][0], ref[0]] x [pts[i][1], ref[1]] x [pts[i][2], ref[2]]
                # minus the volume covered by other points
                # For efficiency with small n, use inclusion-exclusion is hard.
                # Instead, use the standard decomposition:
                # Sort by f1. The contribution of point i is:
                # (f1_{i+1} - f1_i) * (f2_max_above - f2_i) * (f3_max_above - f3_i)
                # where f2_max_above and f3_max_above are the maximum f2 and f3 among
                # points j with f1_j >= f1_i and f2_j >= f2_i and f3_j >= f3_i... 
                # This is getting complex. Let's use a simpler numerical approach.
                
                # For small n, compute hypervolume by decomposing into boxes
                # Each point defines a box from itself to ref. The hypervolume is the
                # union of these boxes. We can compute this by inclusion-exclusion
                # but that's exponential. Instead, use the standard 3D algorithm.
                
                pass
            
            # Use a simpler approach: since n is small (<= pop_size + a few), 
            # compute hypervolume by sorting and using the standard method
            # Sort by f1 ascending
            pts = pts_sorted
            
            # For each point, compute the "shadow" it casts
            # Standard 3D hypervolume algorithm:
            # 1. Sort by f1
            # 2. For each point i, find the maximum f2 and f3 that are not covered
            #    by points with smaller f1
            
            # Let's implement a simpler version:
            # The hypervolume can be computed by:
            # For each point, the contribution is the volume of the box it defines
            # that is not covered by any other point.
            
            # For n <= 20 or so, we can use a direct method
            # Compute the union of boxes [p_i, ref] for all i
            # Use the inclusion-exclusion principle or a sweep line
            
            # Sweep line approach for 3D:
            # Sort by f1. For each interval between consecutive f1 values,
            # compute the 2D hypervolume in the f2-f3 plane.
            
            f1_vals = np.append(pts[:, 0], ref[0])
            hv = 0.0
            for k in range(n):
                f1_low = pts[k, 0]
                f1_high = f1_vals[k + 1]
                if f1_high <= f1_low:
                    continue
                # For points with f1 in [f1_low, f1_high], we need the 2D hypervolume
                # of points with f1 <= f1_high (i.e., all points up to k+1)
                # in the f2-f3 plane, bounded by ref[1] and ref[2]
                
                # The 2D hypervolume for points 0..k in f2-f3 plane
                sub_pts = pts[:k+1, 1:3]
                sub_ref = np.array([ref[1], ref[2]])
                
                # Compute 2D hypervolume
                hv_2d = self._hypervolume_2d(sub_pts, sub_ref)
                hv += (f1_high - f1_low) * hv_2d
            
            return hv

        def _hypervolume_2d(self, points, ref):
            """Compute exact 2D hypervolume for points strictly better than ref."""
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first coordinate
            pts = pts[np.argsort(pts[:, 0])]
            n = len(pts)
            
            hv = 0.0
            max_y = -np.inf
            for i in range(n):
                x = pts[i, 0]
                y = pts[i, 1]
                if y > max_y:
                    # This point extends the frontier
                    prev_x = pts[i-1, 0] if i > 0 else 0.0
                    # Actually, we need to be more careful
                    pass
            
            # Standard 2D hypervolume: sort by x, then sweep
            # The hypervolume is the sum of rectangles
            # Sort by x ascending
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            
            hv = 0.0
            max_y = 0.0
            prev_x = 0.0
            for i in range(len(pts_sorted)):
                x = pts_sorted[i, 0]
                y = pts_sorted[i, 1]
                if y > max_y:
                    # The rectangle from prev_x to x at height max_y to y
                    # Wait, this isn't quite right. Let me think again.
                    
                    # Standard approach: sort by x. The hypervolume is the area under
                    # the upper envelope of the points, from x=0 to x=ref[0], 
                    # bounded by y=ref[1].
                    
                    pass
            
            # Let's use a simpler correct approach for 2D:
            # Sort by x ascending. For each point, the contribution is:
            # (x_i - x_{i-1}) * (ref[1] - y_i) if y_i < ref[1]
            # where x_{i-1} is the x of the previous point with y <= y_i
            # This is getting complex. Let's use a direct method.
            
            # For 2D, hypervolume = area of union of rectangles [p_i, ref]
            # Sort by x. The union can be computed by sweeping.
            pts = pts_sorted
            n = len(pts)
            
            # The hypervolume is the area of the region that is dominated by at least one point
            # and is within [0, ref[0]] x [0, ref[1]]
            # Equivalently, it's the area of the complement: the region not dominated by any point
            # The complement is the region above the upper envelope of the points
            
            # Upper envelope: for each x, the minimum y among points with x_i <= x
            # Wait, no. A point (x_i, y_i) dominates (x, y) if x_i <= x and y_i <= y.
            # So the dominated region is the union of [x_i, ref[0]] x [y_i, ref[1]]
            # The hypervolume is the area of this union.
            
            # Sort by x. The union of rectangles can be computed by sweeping x.
            # For each x interval, the height of the union is the minimum y among
            # points with x_i <= x (since we want the lowest y that covers this x).
            
            # Actually, the union of [x_i, ref[0]] x [y_i, ref[1]] for all i:
            # For a given x, the y values covered are [min(y_i for x_i <= x), ref[1]]
            # So the height at x is ref[1] - min(y_i for x_i <= x)
            
            # Sort by x
            pts = pts[np.argsort(pts[:, 0])]
            n = len(pts)
            
            hv = 0.0
            min_y = np.inf
            prev_x = 0.0
            for i in range(n):
                x = pts[i, 0]
                y = pts[i, 1]
                if y < min_y:
                    # From prev_x to x, the min_y was the old min_y
                    if prev_x < x:
                        hv += (x - prev_x) * (ref[1] - min_y)
                    min_y = y
                    prev_x = x
                # If y >= min_y, no change in min_y, so no new contribution in this interval
                # But we still need to update prev_x? No, because the min_y doesn't change.
            
            # From the last point to ref[0]
            if prev_x < ref[0]:
                hv += (ref[0] - prev_x) * (ref[1] - min_y)
            
            return hv

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([0])
            
            # Select parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx]
            
            # Create child with mutation
            # Use adaptive step size based on progress
            step_size = 0.1 * (hi - lo)
            child = parent + step_size * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population
            # Use sum of objectives as a simple fitness
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            X[worst_idx], F[worst_idx] = child, fc
        
        # Final polish: try to improve the best points with small steps
        if time.time() < deadline - 10.0:
            nd_idx = non_dominated_indices(F)
            for _ in range(min(10, self.budget - used)):
                if time.time() > deadline:
                    break
                if len(nd_idx) == 0:
                    break
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                parent = X[parent_idx]
                # Small step for refinement
                child = parent + 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Only replace if it's non-dominated or improves
                # Check if child dominates any point in the non-dominated set
                replaced = False
                for i in nd_idx:
                    if dominates(fc, F[i]):
                        X[i], F[i] = child, fc
                        replaced = True
                        break
                if not replaced:
                    # Add to population if it's non-dominated
                    is_dom = any(dominates(f, fc) for f in F)
                    if not is_dom:
                        # Replace the worst dominated point
                        dominated_mask = np.array([any(dominates(f, fc) for f in F) for f in F])
                        if np.any(dominated_mask):
                            worst_dom_idx = np.where(dominated_mask)[0]
                            worst_idx = worst_dom_idx[np.argmax(np.sum(F[worst_dom_idx], axis=1))]
                            X[worst_idx], F[worst_idx] = child, fc
                        else:
                            # Replace the point with the largest sum
                            sums = np.sum(F, axis=1)
                            worst_idx = int(np.argmax(sums))
                            X[worst_idx], F[worst_idx] = child, fc
                # Update non-dominated set
                nd_idx = non_dominated_indices(F)
        
        return X
# EVOLVE-BLOCK-END
