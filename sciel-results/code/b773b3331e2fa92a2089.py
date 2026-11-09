"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([1705.0, 12.0, 0.3])

        def dominates(a, b):
            """Check if a dominates b (all objectives <= and at least one <)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return indices of non-dominated points"""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(F[i], F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]

        def compute_hypervolume(F, ref):
            """Compute exact 3D hypervolume for non-dominated points vs reference point"""
            # Only count points strictly better than reference in all objectives
            valid_mask = np.all(F < ref, axis=1)
            F_valid = F[valid_mask]
            if len(F_valid) == 0:
                return 0.0
            
            # Keep only non-dominated points among valid ones
            nd_idx = non_dominated_sort(F_valid)
            F_nd = F_valid[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            
            # 3D hypervolume computation
            # Sort by first objective
            order = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[order]
            
            hv = 0.0
            n = len(F_sorted)
            for i in range(n):
                # For each point, compute its contribution
                # A point (f1, f2, f3) contributes volume:
                # (f1_next - f1) * (f2_next - f2) * (f3_next - f3)
                # where f*_next is the next point's value in each dimension
                # This is an approximation; let's use a proper 3D HV algorithm
                
                # Simpler approach: use the "slicing" method
                # For 3D, we can iterate and compute contributions
                pass
            
            # Use a simpler but correct 3D HV computation
            # Sort by first coordinate
            F_sorted = F_nd[np.argsort(F_nd[:, 0])]
            
            hv = 0.0
            n = len(F_sorted)
            for i in range(n):
                f1, f2, f3 = F_sorted[i]
                # Find the "box" contributed by this point
                # The box extends from (f1, f2, f3) to the next point's boundaries
                # In 3D, this is complex. Let's use a different approach.
                
                # Actually, for 3D HV, we can use the following:
                # For each point, its contribution is the volume of the region
                # that is dominated by this point but not by any other point.
                
                # A simpler method: use the fact that HV = sum of contributions
                # where each contribution is computed by considering the point
                # and its neighbors in the sorted order.
                
                # Let's use a brute-force but correct method for small n
                # which is fine for n <= 200
                pass
            
            # For practical purposes with up to 200 points, let's use a 
            # well-known 3D HV algorithm
            # Here's a correct implementation:
            
            # Sort by first objective
            order = np.argsort(F_nd[:, 0])
            F_s = F_nd[order]
            n = len(F_s)
            
            # For each point i, compute the volume of the box it uniquely dominates
            # A point (x1, x2, x3) uniquely dominates the box:
            # [x1, next_x1) x [x2, next_x2) x [x3, next_x3)
            # where next_xj is the minimum value among all points that have 
            # first coordinate > x1, in dimension j.
            
            hv = 0.0
            for i in range(n):
                x1, x2, x3 = F_s[i]
                
                # Find the next point in the first dimension
                if i + 1 < n:
                    next_x1 = F_s[i + 1, 0]
                else:
                    next_x1 = ref[0]
                
                # For dimensions 2 and 3, we need to find the minimum value
                # among points with first coordinate > x1
                mask = F_s[:, 0] > x1
                if np.any(mask):
                    min_x2 = np.min(F_s[mask, 1])
                    min_x3 = np.min(F_s[mask, 2])
                else:
                    min_x2 = ref[1]
                    min_x3 = ref[2]
                
                # The box is [x1, next_x1) x [x2, min_x2) x [x3, min_x3)
                # But we need to be careful: the box should only extend to
                # the extent that it's not covered by other points.
                
                # Actually, this approach is not quite right. Let me use a 
                # standard 3D HV algorithm.
                
                # Standard approach: 
                # 1. Sort points by first objective
                # 2. For each point, compute the area in the 2D projection
                #    that it uniquely dominates, then multiply by the 
                #    first-objective width.
                
                # For point i, the width in dim 1 is (next_x1 - x1)
                # The area in dims 2-3 is the area of the 2D Pareto front
                # formed by points that have dim1 > x1, projected onto dims 2-3,
                # but only the part that is "new" compared to points with dim1 <= x1.
                
                # This is getting complex. Let's use a simpler but valid method.
                
                # For n <= 200, we can use the following:
                # HV = sum over all subsets... no, that's exponential.
                
                # Let's use the "incremental" approach:
                # Add points one by one and compute the HV increase.
                
                # Actually, for this task, since we're comparing relative scores,
                # we can use an approximate HV or a simpler metric.
                
                # But let's implement a correct 3D HV.
                
                # Here's a correct 3D HV algorithm:
                # 1. Sort points by first coordinate
                # 2. Maintain a data structure for the 2D Pareto front
                # 3. For each point, compute the 2D area it adds, multiply by width
                
                # For simplicity and correctness with n <= 200:
                
                # Use the following method:
                # HV = integral over the 3D space of the indicator function
                # that the point is dominated by at least one Pareto point.
                
                # A practical approach for 3D with small n:
                # 1. Sort by first objective
                # 2. For each point i, the contribution is:
                #    (x1_{i+1} - x1_i) * A_i
                #    where A_i is the 2D area in (x2, x3) space that is 
                #    dominated by point i but not by any point j > i (in sorted order)
                #    and not by any point j < i that has x2_j < x2_i and x3_j < x3_i.
                
                # This is still complex. Let's use a different strategy:
                # Since we're only using HV for selection, not for the final score,
                # we can use a simpler diversity metric or a scalarized objective.
                
                # For now, let's use a simple scalarization: sum of objectives
                # combined with a diversity term.
                
                # Actually, let's just use the sum of objectives as a proxy
                # for HV contribution, since we're minimizing.
                
                # The point with the smallest sum is likely to have high HV contribution.
                # But this doesn't capture diversity well.
                
                # Let's use a proper but simple HV approximation:
                # For each point, compute its "volume" as the product of
                # (ref[j] - F[i][j]) for all j, but only if it's non-dominated.
                # This is an upper bound on its actual contribution.
                
                pass
            
            # Let's implement a correct 3D HV using the standard algorithm:
            
            # Sort by first objective
            order = np.argsort(F_nd[:, 0])
            F_s = F_nd[order]
            n = len(F_s)
            
            # For each point, compute its contribution to HV
            # Using the slicing method:
            # HV = sum_i (x1_{i+1} - x1_i) * Area_i
            # where Area_i is the 2D Pareto front area of points j > i,
            # but clipped to the region dominated by point i.
            
            # For 2D area computation:
            def compute_2d_area(points_2d, ref_2d):
                """Compute 2D hypervolume for a set of points"""
                if len(points_2d) == 0:
                    return 0.0
                # Sort by first coordinate
                order = np.argsort(points_2d[:, 0])
                pts = points_2d[order]
                area = 0.0
                n = len(pts)
                for i in range(n):
                    x1, x2 = pts[i]
                    if i + 1 < n:
                        next_x1 = pts[i + 1, 0]
                    else:
                        next_x1 = ref_2d[0]
                    # The width in x1
                    width = next_x1 - x1
                    if width <= 0:
                        continue
                    # The height in x2 is (ref_2d[1] - x2)
                    # But we need to clip by points to the right
                    # For 2D, the area contributed by point i is:
                    # width * (ref_2d[1] - max(x2, min_x2_right))
                    # where min_x2_right is the minimum x2 among points j > i
                    mask = pts[:, 0] > x1
                    if np.any(mask):
                        min_x2_right = np.min(pts[mask, 1])
                    else:
                        min_x2_right = ref_2d[1]
                    effective_height = ref_2d[1] - max(x2, min_x2_right)
                    if effective_height > 0:
                        area += width * effective_height
                return area
            
            hv = 0.0
            for i in range(n):
                x1, x2, x3 = F_s[i]
                if i + 1 < n:
                    next_x1 = F_s[i + 1, 0]
                else:
                    next_x1 = ref[0]
                width1 = next_x1 - x1
                if width1 <= 0:
                    continue
                
                # Get all points j > i (in sorted order)
                if i + 1 < n:
                    pts_right = F_s[i + 1:]
                else:
                    pts_right = np.empty((0, 3))
                
                # The 2D area is the area in (x2, x3) space dominated by point i
                # but not by any point in pts_right.
                # This is the 2D HV of the set {point_i} union pts_right,
                # minus the 2D HV of pts_right alone.
                
                # Point i in 2D: (x2, x3)
                # Reference in 2D: (ref[1], ref[2])
                
                # 2D HV of {point_i} alone:
                # (ref[1] - x2) * (ref[2] - x3) if x2 < ref[1] and x3 < ref[2]
                
                # 2D HV of pts_right:
                # This is the 2D hypervolume of the points in pts_right.
                
                # The contribution of point i to the 3D HV is:
                # width1 * (2D_HV({point_i} union pts_right) - 2D_HV(pts_right))
                
                # But computing 2D_HV for each i is O(n^2), which is fine for n <= 200.
                
                # Let's compute 2D_HV for pts_right:
                if len(pts_right) > 0:
                    hv_right = compute_2d_area(pts_right[:, 1:3], np.array([ref[1], ref[2]]))
                else:
                    hv_right = 0.0
                
                # 2D HV of {point_i} union pts_right:
                # This is the 2D HV of the set containing point_i and all points in pts_right.
                all_pts_2d = np.vstack([np.array([[x2, x3]]), pts_right[:, 1:3]]) if len(pts_right) > 0 else np.array([[x2, x3]])
                hv_all = compute_2d_area(all_pts_2d, np.array([ref[1], ref[2]]))
                
                area_contribution = hv_all - hv_right
                if area_contribution > 0:
                    hv += width1 * area_contribution
            
            return hv

        # Initialize population with diverse points
        pop_size = min(20, budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop: HV-guided local search
        mutation_scale = 0.1
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Compute HV for the current population
            current_hv = compute_hypervolume(F, ref)
            
            # Select the point with the highest marginal HV contribution
            # Approximate: select the point that, if removed, would decrease HV the most
            # Or: select the point that is most "central" in terms of HV contribution
            
            # Simpler: select a random non-dominated point and mutate it
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([0])
            
            # Select a point to mutate
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx].copy()
            
            # Adaptive mutation
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replace the worst point in the population
            # Use a simple criterion: replace the point with the largest sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Adapt mutation scale
            # If the child is better than the parent, increase mutation
            # Otherwise, decrease it
            if np.sum(fc) < np.sum(F[parent_idx]):
                mutation_scale = min(0.5, mutation_scale * 1.1)
            else:
                mutation_scale = max(0.01, mutation_scale * 0.9)
        
        return X
# EVOLVE-BLOCK-END
