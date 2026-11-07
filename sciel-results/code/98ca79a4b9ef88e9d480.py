# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol quasi-random + hypervolume-contribution local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Sobol quasi-random initial sampling (150 points) ---
        n_sobol = min(150, budget)
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol)
        X_init = lo + (hi - lo) * sobol_pts
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = n_sobol

        # Build initial archive: keep only non-dominated points
        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, archive_F):
            """Check if f dominates any point in archive_F, return list of dominated indices."""
            dom_idx = []
            for j, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    dom_idx.append(j)
            return dom_idx

        # Filter initial points to non-dominated set
        archive_X = []
        archive_F = []
        for i in range(n_sobol):
            if not is_dominated(F_init[i], archive_F):
                # Remove any archive points dominated by this new point
                dom = dominates(F_init[i], archive_F)
                for j in sorted(dom, reverse=True):
                    archive_X.pop(j)
                    archive_F.pop(j)
                archive_X.append(X_init[i].copy())
                archive_F.append(F_init[i].copy())

        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        # --- Hypervolume contribution calculation (3D specific) ---
        def hypervolume_contribution(F, ref_pt):
            """Compute hypervolume contribution of each point in F (3D)."""
            n = len(F)
            if n == 0:
                return np.array([])
            
            # Filter to points strictly better than reference in all objectives
            valid = np.all(F < ref_pt, axis=1)
            if not np.any(valid):
                return np.zeros(n)
            
            F_valid = F[valid]
            n_valid = len(F_valid)
            contrib = np.zeros(n)
            
            if n_valid <= 1:
                if n_valid == 1:
                    idx = np.where(valid)[0][0]
                    contrib[idx] = np.prod(ref_pt - F_valid[0])
                return contrib
            
            # For 3D, compute contribution using inclusion-exclusion
            # Sort by first objective
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            # For each point, compute the box it uniquely contributes
            # A point's contribution is the volume of the box [F_i, ref] minus
            # overlaps with other points that are "better" in at least one dimension
            
            # Efficient 3D HV contribution:
            # For point i, its contribution = volume of region where it is the
            # "worst" point in the dominance order for that region.
            
            # Simpler approach: compute total HV with and without each point
            # This is O(n^2 * log n) for 3D which is fine for small n
            
            def hv_3d(points):
                """Compute 3D hypervolume with reference point."""
                m = len(points)
                if m == 0:
                    return 0.0
                # Sort by first objective
                pts = points[np.argsort(points[:, 0])]
                # For 3D: project to 2D for each "layer"
                # Use sweep line approach
                total = 0.0
                prev_x = pts[0, 0]
                # Actually, let's use a simpler O(n^2) method for small n
                # For each point, compute the box it uniquely covers
                for i in range(m):
                    # Box for point i: [pts[i], ref]
                    # Subtract boxes of points that dominate i
                    box_vol = np.prod(ref_pt - pts[i])
                    # Subtract overlap with other points
                    for j in range(m):
                        if i == j:
                            continue
                        # Overlap of box_i and box_j
                        lo_ov = np.maximum(pts[i], pts[j])
                        hi_ov = np.minimum(ref_pt, ref_pt)  # just ref
                        # Actually overlap is [max(pts[i], pts[j]), ref]
                        if np.all(lo_ov < ref_pt):
                            ov_vol = np.prod(ref_pt - lo_ov)
                        else:
                            ov_vol = 0.0
                        # Inclusion-exclusion is complex, use direct method
                    # This is getting complex, let's use a different approach
                    break
                
                # Use the standard 3D HV algorithm
                # Sort by first objective (ascending = better)
                pts_sorted = points[np.argsort(points[:, 0])]
                total_hv = 0.0
                # For each point in sorted order, compute the 2D HV of the
                # region where this point is the minimum in obj1
                for i in range(m):
                    # Points that have obj1 <= pts_sorted[i, 0] and obj2, obj3 define 2D region
                    # The contribution of point i is the 2D HV of points j where
                    # pts_sorted[j, 0] <= pts_sorted[i, 0] (i.e., j <= i in sorted order)
                    # but we need to be more careful
                    
                    # Actually, standard approach:
                    # Sort by obj1. For each point i, consider all points with obj1 >= pts[i,0]
                    # (worse in obj1) and compute 2D HV in obj2-obj3 space,
                    # then multiply by the "width" in obj1.
                    
                    # Let me just use a brute-force O(n^3) for small n
                    pass
                
                # Brute force for 3D HV:
                # HV = sum over all cells in the partition
                # For small n, we can enumerate all combinations
                # Actually, let's use the recursive approach:
                
                def hv_recursive(pts, ref, dim_idx):
                    """Recursive HV calculation."""
                    if dim_idx == len(ref) - 1:
                        # 1D: HV = max(0, ref[dim] - min(pts[:, dim]))
                        if len(pts) == 0:
                            return 0.0
                        return max(0.0, ref[dim_idx] - np.min(pts[:, dim_idx]))
                    
                    # Sort by current dimension
                    order = np.argsort(pts[:, dim_idx])
                    pts_s = pts[order]
                    
                    total = 0.0
                    prev_val = pts_s[0, dim_idx]
                    # For each "slice" in this dimension
                    for i in range(len(pts_s)):
                        width = ref[dim_idx] - pts_s[i, dim_idx]
                        if width <= 0:
                            break
                        # Points that are "active" in this slice: all points with
                        # pts[j, dim_idx] <= pts_s[i, dim_idx] (i.e., j <= i)
                        # Actually, points with value <= current value are better
                        active_pts = pts_s[:i+1]
                        # Project to remaining dimensions
                        remaining = np.delete(active_pts, dim_idx, axis=1)
                        remaining_ref = np.delete(ref, dim_idx)
                        # But we need to handle the width correctly
                        # The slice is from pts_s[i, dim_idx] to min(pts_s[i+1, dim_idx], ref)
                        if i + 1 < len(pts_s):
                            slice_width = min(pts_s[i+1, dim_idx], ref[dim_idx]) - pts_s[i, dim_idx]
                        else:
                            slice_width = ref[dim_idx] - pts_s[i, dim_idx]
                        
                        if slice_width > 0:
                            hv_rest = hv_recursive(remaining, remaining_ref, dim_idx + 1)
                            total += slice_width * hv_rest
                        else:
                            break
                    
                    return total
                
                return hv_recursive(pts, ref_pt, 0)
            
            # For efficiency with small n, compute contribution by removing each point
            total_hv = hv_3d(F_valid)
            for i in range(n_valid):
                F_without = np.delete(F_valid, i, axis=0)
                hv_without = hv_3d(F_without)
                contrib[np.where(valid)[0][i]] = total_hv - hv_without
            
            return contrib

        # --- Phase 2: Hypervolume-contribution-based local search ---
        remaining = budget - used
        rng = np.random.default_rng(42)
        
        for i in range(remaining):
            if time.time() > deadline - 5:
                break
            if used >= budget:
                break
            
            # Compute hypervolume contributions
            if len(archive_F) > 0:
                contribs = hypervolume_contribution(archive_F, ref)
                # Find the point with lowest contribution
                min_contrib_idx = np.argmin(contribs)
            else:
                min_contrib_idx = 0
            
            # Perturb the lowest-contribution point
            center = archive_X[min_contrib_idx]
            
            # Adaptive step size
            progress = i / max(1, remaining - 1)
            step = 0.10 * (1.0 - 0.6 * progress) + 0.02
            
            # Generate multiple perturbations and pick best
            best_child = None
            best_fc = None
            
            n_perturb = 5
            for p in range(n_perturb):
                child = center + step * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                if used >= budget:
                    break
                
                # Check if this child improves hypervolume
                # Simple check: is it non-dominated?
                if not is_dominated(fc, archive_F):
                    # Compute new HV with this point added
                    new_F = np.vstack([archive_F, fc.reshape(1, -1)])
                    # For efficiency, just check dominance
                    if best_fc is None or np.sum(fc) < np.sum(best_fc):
                        best_fc = fc
                        best_child = child
            
            if best_child is not None:
                # Replace the low-contribution point with the best child
                archive_X[min_contrib_idx] = best_child
                archive_F[min_contrib_idx] = best_fc
                
                # Remove any archive points now dominated
                new_archive_X = [best_child]
                new_archive_F = [best_fc]
                for j in range(len(archive_X)):
                    if j == min_contrib_idx:
                        continue
                    if not (np.all(best_fc <= archive_F[j]) and np.any(best_fc < archive_F[j])):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                
                archive_X = np.array(new_archive_X)
                archive_F = np.array(new_archive_F)

        return archive_X

# EVOLVE-BLOCK-END
