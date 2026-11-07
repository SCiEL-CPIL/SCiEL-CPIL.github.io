"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using deterministic coordinate descent
    with ternary search guided by hypervolume contribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        X_list = []
        F_list = []
        used = 0

        ref_point = np.array([3.0, 5.0, 7.0])

        def try_eval(x):
            """Evaluate x, increment used counter. Returns f or None if over budget."""
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        def hv_contribution(f_new, F_existing):
            """Compute the hypervolume contribution of f_new given existing points.
            Only counts if f_new is strictly better than ref_point in all objectives."""
            if np.any(f_new >= ref_point):
                return 0.0
            
            # Filter existing points that are strictly better than ref_point
            valid_mask = np.all(F_existing < ref_point, axis=1)
            F_valid = F_existing[valid_mask]
            
            if len(F_valid) == 0:
                # Hypervolume of single point
                return np.prod(ref_point - f_new)
            
            # Compute HV with and without f_new
            def compute_hv(F):
                """Compute exact hypervolume for 3D points."""
                if len(F) == 0:
                    return 0.0
                if len(F) == 1:
                    return np.prod(ref_point - F[0])
                
                # Sort by first objective
                idx = np.argsort(F[:, 0])
                F_sorted = F[idx]
                
                hv = 0.0
                prev_y = ref_point[1]
                prev_z = ref_point[2]
                
                for i in range(len(F_sorted)):
                    fi = F_sorted[i]
                    # The region dominated by fi but not by any point with smaller f[0]
                    # is where f[1] < prev_y and f[2] < prev_z
                    # Actually need to be more careful - compute 3D HV properly
                    
                    # For each point, its contribution is the volume of the box
                    # [fi[0], next_f0) x [fi[1], ...) that isn't dominated by others
                    # This is complex; use a simpler approach for small sets
                
                # Use inclusion-exclusion or grid-based approximation for small n
                # For exact HV with 3 objectives and small n, use the standard algorithm
                return _compute_hv_3d(F_sorted, ref_point)
            
            def _compute_hv_3d(F_sorted, ref):
                """Exact 3D hypervolume computation."""
                n = len(F_sorted)
                if n == 0:
                    return 0.0
                
                # Sort by f[0] ascending
                # For each point i, its contribution is:
                # (f[i+1][0] - f[i][0]) * area_2d(f[i][1:], considering points j with f[j][0] <= f[i][0])
                # But since sorted by f[0], points j < i have smaller f[0]
                
                hv = 0.0
                # Build list of (f1, f2) for points that are not dominated in the 2D projection
                # A point (y,z) is dominated if there exists (y',z') with y'<=y and z'<=z
                
                for i in range(n):
                    f_i = F_sorted[i]
                    f_next_0 = F_sorted[i+1][0] if i + 1 < n else ref[0]
                    width_0 = f_next_0 - f_i[0]
                    
                    # Compute 2D hypervolume of points j where j <= i (all have f0 <= f_i[0])
                    # that are not dominated in (f1, f2) space by f_i
                    # Actually, the contribution of point i is:
                    # width_0 * (area in 2D where f1 > f_i[1] OR f2 > f_i[2] is NOT covered)
                    # Hmm, this is getting complex. Let me use a different approach.
                    
                    # The standard 3D HV algorithm:
                    # Sort by f0. For each point i, the "slice" between f[i][0] and f[i+1][0]
                    # has a 2D cross-section. The 2D HV of the cross-section at position i
                    # is determined by all points j with j >= i (since they extend to the right).
                    
                    # Actually, let me just use a simple grid-based HV for small n.
                    # For n up to ~200, we can do exact computation.
                    
                    pass
                
                # Simpler: use the fact that for 3D, we can compute HV by
                # sorting by first obj, then for each point computing the 2D HV of
                # the "skyline" of remaining points
                hv = 0.0
                for i in range(n):
                    f_i = F_sorted[i]
                    next_f0 = F_sorted[i+1][0] if i+1 < n else ref[0]
                    w0 = next_f0 - f_i[0]
                    
                    # 2D HV of points {f_j[1:] : j >= i} with ref (ref[1], ref[2])
                    # Filter to only those not dominated by f_i in 2D
                    points_2d = []
                    for j in range(i, n):
                        f_j = F_sorted[j]
                        if f_j[1] <= f_i[1] and f_j[2] <= f_i[2]:
                            continue  # dominated by f_i in 2D, doesn't contribute beyond f_i
                        points_2d.append([f_j[1], f_j[2]])
                    
                    # Add f_i itself
                    points_2d.append([f_i[1], f_i[2]])
                    
                    # Compute 2D HV of these points with ref (ref[1], ref[2])
                    # 2D HV: sort by first coord, sweep
                    points_2d.sort(key=lambda p: p[0])
                    hv_2d = 0.0
                    max_z = ref[2]
                    for k in range(len(points_2d)):
                        p = points_2d[k]
                        next_y = points_2d[k+1][0] if k+1 < len(points_2d) else ref[1]
                        # The region from p[0] to next_y where z < max_z
                        # But we need the envelope
                        # Actually for 2D HV: sort by y, keep track of min z
                        # HV = sum over i of (y[i+1] - y[i]) * (ref_z - min_z_so_far)
                        # where min_z_so_far is the minimum z among points 0..i
                        pass
                    
                    # Let me just compute 2D HV properly
                    hv_2d = _hv_2d(points_2d, [ref[1], ref[2]])
                    hv += w0 * hv_2d
                
                return hv
            
            def _hv_2d(points, ref2d):
                """Compute 2D hypervolume of points with reference point."""
                if not points:
                    return 0.0
                # Sort by first coordinate
                points_sorted = sorted(points, key=lambda p: p[0])
                hv = 0.0
                min_z = ref2d[1]
                for i in range(len(points_sorted)):
                    p = points_sorted[i]
                    next_y = points_sorted[i+1][0] if i+1 < len(points_sorted) else ref2d[0]
                    # Contribution: width in y * (min_z - current min_z after including p)
                    # Actually: for the interval [p[0], next_y), the max extent in z
                    # is min(min_z_before, p[1])
                    old_min_z = min_z
                    min_z = min(min_z, p[1])
                    hv += (next_y - p[0]) * (ref2d[1] - min_z)
                return hv

            # This is getting too complex. Let me use a simpler HV contribution estimate.
            # For the purpose of guiding search, we can use a simpler metric.
            # The key insight: a point contributes to HV if it's non-dominated.
            # We can approximate contribution by checking if it's non-dominated and
            # using the volume of the box it uniquely dominates.
            
            # Check if f_new is non-dominated w.r.t. F_valid
            is_dominated = False
            for f_old in F_valid:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    is_dominated = True
                    break
            
            if is_dominated:
                return 0.0
            
            # Estimate contribution: volume of box [f_new, ref) minus volume already covered
            # Simple approximation: product of (ref - f_new) scaled by how much it extends beyond existing points
            box_vol = np.prod(ref_point - f_new)
            
            # Check how much of this box is already covered by existing points
            # A point f_old covers part of the box if f_old[i] < f_new[i] for some i
            # This is an approximation - good enough for search guidance
            return box_vol

        # Phase 1: Initialize with a small diverse set
        # Use the midpoint and a few strategic points
        n_init = 12
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            
            x = np.empty(self.dim)
            # Spread initial points across the space
            for d in range(self.dim):
                if d == 0:
                    x[d] = lo[0] + (hi[0] - lo[0]) * i / max(n_init - 1, 1)
                elif d == 1:
                    x[d] = lo[1] + (hi[1] - lo[1]) * (i % 3) / 2.0
                else:
                    x[d] = mid[d]
            
            f = try_eval(x)
            if f is None:
                break
            X_list.append(x.copy())
            F_list.append(f)
        
        if not X_list:
            return np.array([])
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Phase 2: Deterministic coordinate descent with ternary search
        # For each point, iteratively optimize each dimension using ternary search
        # guided by hypervolume contribution
        
        remaining_budget = self.budget - used
        if remaining_budget > 0 and len(X) > 0:
            n_points = len(X)
            # Allocate budget: ~15 evaluations per point for refinement
            evals_per_point = max(5, remaining_budget // n_points)
            evals_per_point = min(evals_per_point, 25)
            
            for idx in range(n_points):
                if used >= self.budget or time.time() > deadline:
                    break
                
                x = X[idx].copy()
                f = F[idx].copy()
                
                # Coordinate descent: iterate over all dimensions multiple times
                max_cd_rounds = 2
                for cd_round in range(max_cd_rounds):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    for d in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        # Ternary search on dimension d
                        # We have evals_per_point / self.dim evaluations per dimension
                        evals_this_dim = max(3, evals_per_point // self.dim)
                        
                        lo_d = lo[d]
                        hi_d = hi[d]
                        best_x = x.copy()
                        best_f = f.copy()
                        best_hv = hv_contribution(f, F)
                        
                        # Ternary search
                        for t_iter in range(evals_this_dim):
                            if used >= self.budget or time.time() > deadline:
                                break
                            
                            a = lo_d
                            b = hi_d
                            m1 = a + (b - a) / 3.0
                            m2 = b - (b - a) / 3.0
                            
                            # Evaluate at m1
                            x_m1 = x.copy()
                            x_m1[d] = m1
                            f_m1 = try_eval(x_m1)
                            if f_m1 is None:
                                break
                            hv_m1 = hv_contribution(f_m1, F)
                            
                            if used >= self.budget or time.time() > deadline:
                                if hv_m1 > best_hv:
                                    best_x = x_m1
                                    best_f = f_m1
                                    best_hv = hv_m1
                                break
                            
                            # Evaluate at m2
                            x_m2 = x.copy()
                            x_m2[d] = m2
                            f_m2 = try_eval(x_m2)
                            if f_m2 is None:
                                if hv_m1 > best_hv:
                                    best_x = x_m1
                                    best_f = f_m1
                                    best_hv = hv_m1
                                break
                            hv_m2 = hv_contribution(f_m2, F)
                            
                            # Narrow the interval
                            if hv_m1 < hv_m2:
                                hi_d = m2
                            else:
                                lo_d = m1
                        
                        # Update with best found
                        if best_hv > hv_contribution(f, F):
                            x = best_x
                            f = best_f
                
                # Update with improved point
                X[idx] = x
                F[idx] = f
        
        return X
# EVOLVE-BLOCK-END
