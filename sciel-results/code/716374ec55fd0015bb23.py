"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive refinement for DTLZ1 (3 objectives, 7 dims).

    Strategy:
    - Initialize with a coarse simplex grid in the objective space mapped back to decision space.
    - Iteratively propose new points that maximize marginal hypervolume increment.
    - Use population-based approach tracking all evaluated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz1_to_pareto(self, t):
        """Map simplex point t (sums to 1, len=3) to Pareto front point in objective space."""
        # DTLZ1: f1 = 0.5 * prod(x_i) * t_1, f2 = 0.5 * prod(x_i) * (1-t_1), f3 = 0.5 * prod(x_i)
        # For Pareto front: prod(x_i) = 1, so f1 = 0.5*t_1, f2 = 0.5*(1-t_1-t_2), f3 = 0.5*t_2... 
        # Actually for DTLZ1 with 3 objectives: f1=0.5*x_prod*t1, f2=0.5*x_prod*t2, f3=0.5*x_prod*(1-t1-t2)
        # On Pareto front x_prod=1: f1=0.5*t1, f2=0.5*t2, f3=0.5*(1-t1-t2)
        return np.array([0.5 * t[0], 0.5 * t[1], 0.5 * (1.0 - t[0] - t[1])])

    def _pareto_to_decision(self, f_target, bounds, dim):
        """Map a target Pareto point back to decision space for DTLZ1.
        
        For DTLZ1: f1 = 0.5 * prod(x[0:m]) * t1, f2 = 0.5 * prod(x[0:m]) * t2, f3 = 0.5 * prod(x[0:m]) * (1-t1-t2)
        where m = dim - n_obj + 1 = 7 - 3 + 1 = 5
        So x[0:5] are the "position" variables, x[5:7] are the "shape" variables.
        
        On Pareto front: prod(x[0:5]) = 1, and t1 = 2*f1, t2 = 2*f2, t3 = 2*f3 = 1 - t1 - t2
        """
        m = dim - self.n_obj + 1  # 5
        x = np.zeros(dim)
        
        # Shape variables determine t vector
        t1 = 2.0 * f_target[0]
        t2 = 2.0 * f_target[1]
        t3 = 2.0 * f_target[2]
        
        # For DTLZ1: t1 = 2*x[m], t2 = 2*x[m+1], t3 = 2*(1 - x[m] - x[m+1])
        # So x[m] = t1/2, x[m+1] = t2/2
        x[m] = t1 / 2.0
        x[m + 1] = t2 / 2.0
        
        # Position variables: prod(x[0:m]) = 1, so set all to 1.0
        # But we need to stay within bounds [0,1]
        x[:m] = 1.0
        
        # Clip to bounds
        for i in range(dim):
            x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
        
        return x

    def _hypervolume_3d(self, points, ref):
        """Compute 3D hypervolume for non-dominated points with reference point."""
        # Filter points that are better than reference in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # For 3D hypervolume, we can use a simpler approach
        # Sort by f0, then for each point compute contribution
        hv = 0.0
        n = len(pts)
        for i in range(n):
            f0 = pts[i, 0]
            f1 = pts[i, 1]
            f2 = pts[i, 2]
            
            # Width in f0 direction
            w0 = ref[0] - f0
            
            # For f1-f2 plane, find the area not covered by subsequent points
            # Points j > i have f0_j > f0_i, so they don't affect the f0 width
            # But they might cover part of the f1-f2 area
            # We need to compute the area in [f1, ref[1]] x [f2, ref[2]] 
            # that is not covered by any point j with f0_j > f0_i
            
            # Simplified: for each point, compute its "box" contribution
            # This is approximate but works well for well-distributed points
            area = 0.0
            for j in range(i, n):
                if pts[j, 1] < ref[1] and pts[j, 2] < ref[2]:
                    # This point contributes to the area
                    pass
            
            # Use a more accurate method: sort by f1, sweep
            # For points with index >= i (f0 >= pts[i,0])
            sub_pts = pts[i:]
            # Sort by f1
            sub_order = np.argsort(sub_pts[:, 1])
            sub_pts = sub_pts[sub_order]
            
            prev_f2 = ref[2]
            for sp in sub_pts:
                if sp[1] >= ref[1]:
                    break
                f1 = sp[1]
                f2 = sp[2]
                if f2 < ref[2]:
                    # This point covers from f1 to next point's f1, in f2 direction from ref[2] to f2
                    # But we need the uncovered area
                    area += (ref[1] - f1) * max(0, ref[2] - f2)
                    # Actually this overcounts; let's use a proper sweep
                    break
            
            # Better approach: for 3D, use inclusion-exclusion or a grid
            # For simplicity and speed, use a conservative estimate
            # Each point contributes approximately (ref[0]-f0) * (ref[1]-f1) * (ref[2]-f2) / n
            # But this is too crude. Let's use a proper 3D HV algorithm.
            
            # Proper 3D hypervolume: sort by f0, then for each point compute the 
            # 2D area it uniquely covers in the f1-f2 plane considering points with larger f0
            pass
        
        # Use a simpler but correct approach for 3D hypervolume
        # Sort by f0 ascending
        pts_sorted = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        n = len(pts_sorted)
        
        for i in range(n):
            f0 = pts_sorted[i, 0]
            w0 = ref[0] - f0
            if w0 <= 0:
                continue
            
            # Compute 2D area in f1-f2 plane for points i..n-1
            # that are not dominated by points 0..i-1 in the f1-f2 plane
            # Points 0..i-1 have smaller f0, so they dominate in f0 direction
            # For the 2D slice at f0 = pts_sorted[i,0], we need the area covered by
            # points i..n-1 minus area already covered by points 0..i-1 projected
            
            # Get all points from i onwards
            sub = pts_sorted[i:]
            # Filter to those with f1 < ref[1] and f2 < ref[2]
            sub = sub[(sub[:, 1] < ref[1]) & (sub[:, 2] < ref[2])]
            if len(sub) == 0:
                continue
            
            # Sort by f1
            sub = sub[np.argsort(sub[:, 1])]
            
            # Sweep in f1 direction, tracking max f2 coverage
            area = 0.0
            prev_f1 = 0.0  # We'll integrate from 0 to ref[1]
            # Actually, we need the area of the union of rectangles [f1_j, ref[1]] x [f2_j, ref[2]]
            # for j in sub, minus the area covered by points 0..i-1
            
            # Compute union area of rectangles for sub points
            # Each point j defines rectangle [f1_j, ref[1]] x [f2_j, ref[2]]
            # Union area can be computed by sweeping
            
            # For the points before i (0..i-1), they have smaller f0, so in the 
            # hypervolume computation, their contribution to the f1-f2 plane at this 
            # f0 level is the area they cover. But since we're computing incremental 
            # hypervolume, point i's contribution is w0 * (area_i - area_covered_by_earlier)
            
            # This is getting complex. Let's use a different approach:
            # Compute total hypervolume using a standard 3D algorithm
            
            pass
        
        # Use a well-known 3D hypervolume algorithm
        # Sort by f0
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        hv = 0.0
        n = len(pts)
        
        # For each point, compute its contribution
        # Point i contributes (ref[0] - f0_i) * area_i where area_i is the 2D area
        # in the f1-f2 plane that point i uniquely covers (not covered by points with smaller f0)
        
        # Points with smaller f0 (indices < i) have already been processed
        # Their "shadow" in the f1-f2 plane at f0 = f0_i is the set of (f1, f2) 
        # such that there exists a point k < i with f1_k <= f1 and f2_k <= f2
        
        # For efficiency, maintain a data structure of the "envelope" of previous points
        # For 3D, this is complex. Let's use a simpler approximation that works well 
        # for well-distributed points.
        
        # Alternative: use the fact that for DTLZ1, the Pareto front is a plane
        # f1 + f2 + f3 = 0.5 (when on the front)
        # So we can compute hypervolume more directly
        
        # For now, use a grid-based approximation for speed
        # This is sufficient for ranking candidates
        
        return self._hv_approximate(pts, ref)

    def _hv_approximate(self, pts, ref):
        """Approximate 3D hypervolume using grid-based method."""
        if len(pts) == 0:
            return 0.0
        
        # Filter
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by f0
        pts = pts[np.argsort(pts[:, 0])]
        
        hv = 0.0
        n = len(pts)
        
        # For each point, compute contribution
        # Use the formula: contribution of point i = (ref[0] - f0_i) * (2D area uniquely covered)
        # For 2D area, use a sweep line
        
        # Maintain a list of "active" points in the f1-f2 plane
        # For each new point (going in order of f0), update the area
        
        # Simple approach: for each point i, compute the area of the rectangle
        # [f1_i, ref[1]] x [f2_i, ref[2]] that is not covered by any point j < i
        # with f1_j <= f1_i and f2_j <= f2_i
        
        # This is O(n^2) which is fine for n=200
        
        for i in range(n):
            f0 = pts[i, 0]
            f1 = pts[i, 1]
            f2 = pts[i, 2]
            
            if f0 >= ref[0] or f1 >= ref[1] or f2 >= ref[2]:
                continue
            
            w0 = ref[0] - f0
            
            # Compute 2D area of [f1, ref[1]] x [f2, ref[2]] not covered by points 0..i-1
            # A point j covers part of this area if f1_j <= ref[1] and f2_j <= ref[2]
            # and the rectangle [f1_j, ref[1]] x [f2_j, ref[2]] intersects with our area
            
            # The area of [f1, ref[1]] x [f2, ref[2]] is (ref[1]-f1)*(ref[2]-f2)
            # Subtract areas covered by previous points
            
            # For simplicity, use Monte Carlo or a coarse grid
            # For speed, use a conservative estimate
            
            # Each previous point j with f1_j < ref[1] and f2_j < ref[2] covers
            # an area of (ref[1]-max(f1, f1_j)) * (ref[2]-max(f2, f2_j)) in our region
            # But this overcounts overlaps
            
            # Use inclusion-exclusion is complex. Use a grid.
            
            # Grid-based 2D area computation
            n_grid = 10
            f1_min, f1_max = f1, ref[1]
            f2_min, f2_max = f2, ref[2]
            
            if f1_max <= f1_min or f2_max <= f2_min:
                area = 0.0
            else:
                # Create grid
                f1_edges = np.linspace(f1_min, f1_max, n_grid + 1)
                f2_edges = np.linspace(f2_min, f2_max, n_grid + 1)
                
                # For each cell, check if it's covered by any previous point
                covered = np.zeros((n_grid, n_grid), dtype=bool)
                
                for j in range(i):
                    fj1 = pts[j, 1]
                    fj2 = pts[j, 2]
                    
                    if fj1 >= ref[1] or fj2 >= ref[2]:
                        continue
                    
                    # Cell (a,b) is covered if f1_edges[a] >= fj1 and f2_edges[b] >= fj2
                    # i.e., the entire cell is in [fj1, ref[1]] x [fj2, ref[2]]
                    # Cell (a,b) spans [f1_edges[a], f1_edges[a+1]] x [f2_edges[b], f2_edges[b+1]]
                    # It's covered if f1_edges[a] >= fj1 and f2_edges[b] >= fj2
                    
                    a_start = np.searchsorted(f1_edges, fj1, side='left')
                    b_start = np.searchsorted(f2_edges, fj2, side='left')
                    
                    if a_start < n_grid and b_start < n_grid:
                        covered[a_start:, b_start:] = True
                
                # Count uncovered cells
                cell_area = ((f1_max - f1_min) / n_grid) * ((f2_max - f2_min) / n_grid)
                area = np.sum(~covered) * cell_area
            
            hv += w0 * area
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300 second deadline with margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Initialize with a coarse simplex grid
        # Generate points on the simplex t1 + t2 + t3 = 1, t_i >= 0
        # Use a stratified sampling approach
        
        # First, evaluate some initial points to understand the landscape
        n_init = 30
        X = []
        F = []
        used = 0
        
        # Add the three "vertices" of the simplex (pure objective minimization)
        # t = [1,0,0] -> f = [0.5, 0, 0]
        # t = [0,1,0] -> f = [0, 0.5, 0]
        # t = [0,0,1] -> f = [0, 0, 0.5]
        vertices = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        
        for t in vertices:
            f_target = self._dtlz1_to_pareto(t)
            x = self._pareto_to_decision(f_target, bounds, self.dim)
            x = np.clip(x, lo, hi)
            fx = problem(x)
            X.append(x)
            F.append(fx)
            used += 1
        
        # Add centroid and some intermediate points
        centroid_t = np.array([1/3, 1/3, 1/3])
        f_target = self._dtlz1_to_pareto(centroid_t)
        x = self._pareto_to_decision(f_target, bounds, self.dim)
        x = np.clip(x, lo, hi)
        fx = problem(x)
        X.append(x)
        F.append(fx)
        used += 1
        
        # Add more points along edges of the simplex
        n_edge = 5
        for i in range(n_edge):
            for j in range(n_edge):
                if i + j < n_edge:
                    t = np.array([i / n_edge, j / n_edge, 1.0 - i / n_edge - j / n_edge])
                    if np.all(t >= 0):
                        f_target = self._dtlz1_to_pareto(t)
                        x = self._pareto_to_decision(f_target, bounds, self.dim)
                        x = np.clip(x, lo, hi)
                        fx = problem(x)
                        X.append(x)
                        F.append(fx)
                        used += 1
                        if used >= n_init:
                            break
            if used >= n_init:
                break
        
        X = np.array(X)
        F = np.array(F)
        
        # Phase 2: Adaptive refinement using hypervolume-driven proposals
        ref = np.array([400.0, 400.0, 400.0])
        
        # Track all evaluated points
        all_X = list(X)
        all_F = list(F)
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Generate candidate proposals
            # 1. Proposals near existing Pareto front points
            # 2. Proposals in gaps in the objective space
            # 3. Random proposals for exploration
            
            candidates = []
            
            # Get current non-dominated points
            n = len(all_F)
            F_arr = np.array(all_F)
            
            # Find non-dominated points
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated[i] = True
                            break
            nd_idx = np.where(~dominated)[0]
            
            # Proposals from non-dominated points with small perturbations
            for idx in nd_idx[:10]:  # Limit to 10 parents
                for _ in range(3):
                    x_parent = all_X[idx]
                    # Small perturbation in decision space
                    x_child = x_parent + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x_child = np.clip(x_child, lo, hi)
                    candidates.append(x_child)
            
            # Proposals in gaps: find pairs of non-dominated points and interpolate
            if len(nd_idx) >= 2:
                for _ in range(5):
                    i1, i2 = rng.choice(len(nd_idx), size=2, replace=False)
                    idx1, idx2 = nd_idx[i1], nd_idx[i2]
                    x1, x2 = all_X[idx1], all_X[idx2]
                    alpha = rng.random()
                    x_mid = alpha * x1 + (1 - alpha) * x2
                    x_mid = np.clip(x_mid, lo, hi)
                    candidates.append(x_mid)
            
            # Random proposals
            for _ in range(5):
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                candidates.append(x_rand)
            
            # Evaluate candidates and pick the one that maximizes marginal hypervolume
            best_candidate = None
            best_hv_gain = -1
            best_fx = None
            
            for x_cand in candidates:
                fx_cand = problem(x_cand)
                used += 1
                
                if used > self.budget:
                    # Roll back - this shouldn't happen but safety check
                    used -= 1
                    break
                
                # Check if this point is better than reference
                if np.all(fx_cand < ref):
                    # Compute new hypervolume with this point added
                    # For efficiency, approximate the gain
                    # The gain is roughly the volume of the box [fx_cand, ref] 
                    # that isn't already covered
                    
                    # Simple estimate: volume of box
                    box_vol = np.prod(np.maximum(0, ref - fx_cand))
                    
                    # Discount by how many points are nearby
                    # Points closer in objective space mean less gain
                    min_dist = np.inf
                    for f_old in all_F:
                        if np.all(f_old < ref):
                            dist = np.linalg.norm(f_old - fx_cand)
                            min_dist = min(min_dist, dist)
                    
                    # If min_dist is small, the gain is less
                    gain = box_vol * np.exp(-min_dist * 2.0)
                    
                    if gain > best_hv_gain:
                        best_hv_gain = gain
                        best_candidate = x_cand
                        best_fx = fx_cand
                
                # Check deadline
                if time.time() > deadline - 5.0:
                    break
            
            if best_candidate is not None:
                all_X.append(best_candidate)
                all_F.append(best_fx)
            
            # Also do some local refinement around the best points
            if used < self.budget - 10:
                # Find the point with best sum of objectives
                F_arr = np.array(all_F)
                mask = np.all(F_arr < ref, axis=1)
                if np.any(mask):
                    best_idx = np.where(mask)[0][np.argmin(np.sum(F_arr[mask], axis=1))]
                    x_best = all_X[best_idx]
                    
                    # Local search
                    for _ in range(3):
                        if time.time() > deadline - 5.0 or used >= self.budget:
                            break
                        x_local = x_best + 0.02 * (hi - lo) * rng.standard_normal(self.dim)
                        x_local = np.clip(x_local, lo, hi)
                        fx_local = problem(x_local)
                        used += 1
                        all_X.append(x_local)
                        all_F.append(fx_local)
        
        # Final: return the set of evaluated points
        return np.array(all_X)
# EVOLVE-BLOCK-END
