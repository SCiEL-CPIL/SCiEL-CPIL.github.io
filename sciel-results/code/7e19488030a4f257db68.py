"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using HV-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Helper: Check if point a dominates point b
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        # Helper: Compute 2D Hypervolume for a set of points
        def hv_2d(points, ref_2d):
            if len(points) == 0:
                return 0.0
            # Sort by first coordinate
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            hv = 0.0
            for i in range(len(pts)):
                x1, x2 = pts[i]
                if x1 >= ref_2d[0] or x2 >= ref_2d[1]:
                    continue
                # Width in x1 direction
                if i + 1 < len(pts):
                    next_x1 = pts[i + 1, 0]
                else:
                    next_x1 = ref_2d[0]
                width = next_x1 - x1
                if width <= 0:
                    continue
                # Height in x2 direction
                # The height is determined by the minimum x2 of all points to the right (i+1 to end)
                # Actually, for 2D, the contribution of point i is width * (ref_y - max(x2_i, min_x2_right))
                # Wait, standard 2D HV: 
                # Sort by x1 ascending.
                # For each point, the box it uniquely dominates is [x1_i, x1_{i+1}) x [x2_i, ref_y) 
                # BUT ONLY IF x2_i is less than all x2_j for j > i.
                # If there is a point j > i with x2_j < x2_i, then the height is limited by x2_j?
                # No. The union of boxes.
                # Let's use the standard algorithm:
                # Sort by x1.
                # Maintain a running minimum of x2 from right to left?
                # Actually, simpler:
                # The HV is the sum of areas of rectangles.
                # For a sorted list (x1, x2), the area contributed by point i is:
                # (x1_{i+1} - x1_i) * (ref_y - x2_i) 
                # ONLY IF x2_i is the minimum x2 among points i..end.
                # If x2_i is not the minimum, it is covered by a point to its right with smaller x2.
                
                # Let's re-implement 2D HV correctly.
                # Sort by x1 ascending.
                # Iterate from right to left, keeping track of min_x2_seen.
                # But we need to calculate the contribution of each point.
                
                # Alternative: 
                # HV = sum_{i} (x1_{i+1} - x1_i) * (ref_y - min(x2_i, min_{j>i} x2_j))
                # This is not quite right either.
                
                # Correct 2D HV:
                # Sort points by x1.
                # For each point i, let x1_next be x1 of point i+1 (or ref_x).
                # Let x2_eff be the minimum x2 among points i, i+1, ..., end.
                # Then the vertical extent of the region dominated by the "front" up to x1_next is x2_eff.
                # The area added in the slice [x1_i, x1_next] is (x1_next - x1_i) * (ref_y - x2_eff).
                # Wait, this calculates the area under the lower envelope.
                
                # Let's use the incremental method which is safer for small n.
                # Or just brute force for 2D if n is small?
                # n is at most 200. 2D HV is O(n log n).
                
                # Let's stick to the slice method:
                # 1. Sort by x1.
                # 2. Pre-compute min_x2_from_right[i] = min(x2[i], min_x2_from_right[i+1])
                # 3. For each i, width = x1[i+1] - x1[i] (or ref_x - x1[i] for last)
                # 4. height = ref_y - min_x2_from_right[i]
                # 5. area += width * height (if height > 0)
                
                min_x2_from_right = np.empty(len(pts))
                min_x2_from_right[-1] = pts[-1, 1]
                for i in range(len(pts) - 2, -1, -1):
                    min_x2_from_right[i] = min(pts[i, 1], min_x2_from_right[i + 1])
                
                for i in range(len(pts)):
                    x1 = pts[i, 0]
                    if i + 1 < len(pts):
                        x1_next = pts[i + 1, 0]
                    else:
                        x1_next = ref_2d[0]
                    width = x1_next - x1
                    if width <= 0:
                        continue
                    x2_eff = min_x2_from_right[i]
                    if x2_eff >= ref_2d[1]:
                        continue
                    height = ref_2d[1] - x2_eff
                    hv += width * height
            return hv

        # Helper: Compute 3D Hypervolume
        def hv_3d(points, ref_3d):
            if len(points) == 0:
                return 0.0
            # Filter valid points
            valid_mask = np.all(points < ref_3d, axis=1)
            pts = points[valid_mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by x1
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            n = len(pts)
            
            hv = 0.0
            # For each point i, the contribution is (x1_{i+1} - x1_i) * HV_2D(points_i_to_end in x2,x3)
            # Wait, this is not correct.
            # The correct slicing method for 3D:
            # HV = sum_{i} (x1_{i+1} - x1_i) * Area_2D( { (x2_j, x3_j) for j >= i } )
            # Where Area_2D is the 2D hypervolume of the set of points from i to end, 
            # projected onto x2-x3 plane, with reference (ref_x2, ref_x3).
            
            # We can compute this efficiently by iterating i from n-1 down to 0,
            # maintaining the 2D HV of the set {i, ..., n-1}.
            # But adding a point to the 2D set and recalculating HV is O(n log n).
            # Total O(n^2 log n). For n=200, this is 200*200*8 ~ 320k ops. Fast enough.
            
            # Let's just compute HV_2D for each suffix.
            # To be faster, we can just call hv_2d on the suffix.
            
            for i in range(n):
                x1 = pts[i, 0]
                if i + 1 < n:
                    x1_next = pts[i + 1, 0]
                else:
                    x1_next = ref_3d[0]
                width = x1_next - x1
                if width <= 0:
                    continue
                
                # Get points from i to end
                suffix = pts[i:, 1:3]
                area = hv_2d(suffix, ref_3d[1:3])
                hv += width * area
                
            return hv

        # Initialize population
        # Use Latin Hypercube Sampling for diversity
        pop_size = 20
        if budget < pop_size:
            pop_size = budget
            
        # Simple random initialization for now, LHS is complex to implement quickly
        X = np.array([lo + (hi - lo) * np.random.default_rng(i).random(dim) for i in range(pop_size)])
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        # Main Loop: Greedy HV Local Search
        # We will try to improve the HV by refining points.
        # Strategy:
        # 1. Calculate current HV.
        # 2. For each point, calculate its marginal contribution (HV with point vs HV without).
        #    Actually, removing a point might not decrease HV if it's dominated.
        #    We only care about non-dominated points.
        # 3. Pick the point with the highest marginal contribution.
        # 4. Try to move it slightly in each dimension to increase HV.
        
        # To save time, we will not calculate marginal HV for all points every step.
        # We will just pick a random non-dominated point and try to improve it.
        # Or, we can keep a list of "promising" points.
        
        # Let's use a simpler strategy:
        # Iterate through points. For each point, try small steps in +/- direction for each dimension.
        # If a step improves the total HV (by replacing the point), accept it.
        
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005]
        
        # We have budget - pop_size evaluations left.
        # Each "improvement attempt" costs 1 evaluation.
        # We will cycle through points and step sizes.
        
        # To be efficient, we will only evaluate if the new point is non-dominated or improves HV.
        # But we don't know HV until we evaluate.
        
        # Let's use a queue of (point_index, step_size_idx, dim_idx, direction).
        # This is getting complex.
        
        # Simpler: 
        # While evals_used < budget:
        #   Pick a random point from current population.
        #   Pick a random step size.
        #   Pick a random dimension and direction.
        #   Evaluate new point.
        #   If new point improves the total HV (replacing the original), accept.
        #   Else, reject.
        
        # This is basically random local search with HV acceptance.
        
        rng = np.random.default_rng(42)
        
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Pick a parent
            parent_idx = rng.integers(pop_size)
            parent = X[parent_idx].copy()
            
            # Pick a mutation
            # Use a larger step for exploration, smaller for exploitation
            # Adaptive: if we haven't improved in a while, increase step?
            # For now, fixed schedule.
            step_idx = min(int((budget - evals_used) / 20), len(step_sizes) - 1)
            step = step_sizes[step_idx]
            
            dim_idx = rng.integers(dim)
            direction = rng.choice([-1, 1])
            
            child = parent.copy()
            child[dim_idx] += direction * step * (hi[dim_idx] - lo[dim_idx])
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            evals_used += 1
            
            # Check if this improves HV
            # We need to calculate HV with child replacing parent.
            # This is expensive O(n^2).
            # Optimization: Only calculate if child is non-dominated in the current set?
            # Even if non-dominated, it might not increase HV if it's in a crowded region.
            
            # Heuristic: Accept if child dominates parent or is non-dominated and closer to origin?
            # Let's just calculate HV.
            
            # Save old F
            F_old = F[parent_idx].copy()
            F[parent_idx] = fc
            new_hv = hv_3d(F, ref)
            
            # We need old_hv. We can cache it.
            # Let's maintain current_hv.
            # But calculating current_hv at start of loop is expensive.
            # Let's calculate it once.
            
            # Actually, let's just compare to a cached current_hv.
            # We need to update current_hv when we accept.
            
            # Let's initialize current_hv
            if not hasattr(self, '_current_hv'):
                self._current_hv = hv_3d(F, ref)
            
            # Wait, self._current_hv might be from a previous run? No, __call__ is fresh.
            # But we didn't set it yet.
            
            # Let's restructure:
            # 1. Init F, X.
            # 2. current_hv = hv_3d(F, ref)
            # 3. Loop:
            #    Generate child.
            #    Temporarily replace F[parent_idx] with fc.
            #    new_hv = hv_3d(F, ref)
            #    If new_hv > current_hv:
            #        Accept: X[parent_idx] = child, F[parent_idx] = fc, current_hv = new_hv
            #    Else:
            #        Reject: F[parent_idx] = F_old
            
            # This requires 2 HV calculations per eval (one for new, one for old if we don't cache).
            # But we can just calculate new_hv and compare to current_hv.
            # If we reject, we don't need to recalculate current_hv.
            
            # So:
            # 1. Init F, X.
            # 2. current_hv = hv_3d(F, ref)
            # 3. Loop:
            #    Generate child.
            #    F[parent_idx] = fc
            #    new_hv = hv_3d(F, ref)
            #    If new_hv > current_hv + 1e-9:
            #        Accept: X[parent_idx] = child, current_hv = new_hv
            #    Else:
            #        Reject: F[parent_idx] = F_old
            
            # This is O(n^2) per eval. n=200. 40k ops per eval. 200 evals -> 8M ops. Fast.
            
            # Let's implement this.
            
            # We need to handle the first iteration where current_hv is not set.
            
            if 'current_hv' not in locals():
                current_hv = hv_3d(F, ref)
            
            # Temporarily replace
            F[parent_idx] = fc
            new_hv = hv_3d(F, ref)
            
            if new_hv > current_hv + 1e-9:
                # Accept
                X[parent_idx] = child
                current_hv = new_hv
            else:
                # Reject
                F[parent_idx] = F_old
        
        return X

# EVOLVE-BLOCK-END
