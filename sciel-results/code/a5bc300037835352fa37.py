"""Multi-objective optimizer on RE34 with hypervolume-driven search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE34 (5 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit with 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        def is_better_than_ref(f):
            return np.all(f < ref)

        def hv_contrib(points, ref_point):
            """Compute hypervolume contribution of a set of points (all must be < ref in all dims)."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points that are strictly better than ref
            mask = np.all(pts < ref_point, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # For 3D hypervolume, use inclusion-exclusion
            # Sort by first coordinate
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Incremental hypervolume computation for 3D
            hv = 0.0
            for i in range(len(pts)):
                p = pts[i]
                # Volume of box from p to ref, minus overlap with previously counted
                # Simple approach: for each point, compute the box [p, ref] and subtract overlaps
                # This is O(n^2) but n is small
                box_vol = np.prod(ref_point - p)
                # Subtract overlaps with later points (in sorted order, later points have larger x)
                # Actually, let's use a simpler incremental approach
                # For 3D: hv = sum over points of (ref - p) product, but need to handle overlaps
                # Better: use the standard 3D hypervolume algorithm
                pass
            
            # Use a proper 3D hypervolume computation
            # Sort by x
            pts_sorted = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # The contribution of point i is the volume of the box [p, ref] 
                # minus the volume already covered by points j > i that dominate in y,z
                # In 3D, for point i, the new volume is:
                # (ref_x - p_x) * area_2d of the region in yz-plane not covered by projections of points j>i
                # This gets complex. Let's use a simpler but correct approach.
                
            # Let's use a different approach: enumerate all points and compute HV using the 
            # standard algorithm for small n
            return self._hv_3d(pts, ref_point)

        def _hv_3d(points, ref_point):
            """Compute 3D hypervolume for a set of points, all < ref_point."""
            n = len(points)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref_point - points[0]))
            # Sort by first coordinate
            pts = points[np.argsort(points[:, 0])]
            hv = 0.0
            for i in range(n):
                p = pts[i]
                # Box volume from p to ref
                vol = (ref_point[0] - p[0]) * (ref_point[1] - p[1]) * (ref_point[2] - p[2])
                # Subtract overlap with points j > i
                # For each j > i, the overlap is the box [max(p, pts[j]), ref]
                # But we need to be careful about double counting
                # Standard 3D HV: process points in order, for each point compute the 
                # "new" volume it adds
                # The new volume for point i is the volume of [p_i, ref] minus the volume 
                # of the union of [max(p_i, p_j), ref] for all j > i
                # This is still complex. Let's use a simpler O(n^3) approach for small n.
                
                # Actually, for n <= ~50, we can use the following:
                # HV = sum over all subsets... no, that's exponential.
                
                # Let's use the standard incremental approach for 3D:
                # After sorting by x, for point i, the contribution is:
                # (ref_x - x_i) * H_2D({(y_j, z_j) for j >= i}, (ref_y, ref_z))
                # where H_2D is the 2D hypervolume of the projected points
                # But we need to only count the part not covered by previous points.
                
                # Simpler: just compute total HV directly using inclusion-exclusion for small n
                # For n <= 20, we can use the following:
                # Sort points, then for each point, compute the volume of the box it uniquely contributes.
                
                pass
            
            # Let me implement a clean 3D hypervolume
            return self._compute_hv_3d(pts, ref_point)

        def _compute_hv_3d(pts, ref_point):
            """Compute 3D hypervolume. pts is sorted by x."""
            n = len(pts)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref_point - pts[0]))
            # Recursively: the hypervolume is the sum over each point of the volume
            # of the box it uniquely defines
            # For 3D, after sorting by x:
            # HV = sum_i (ref_x - x_i) * H2({(y_j, z_j) : j >= i}, (ref_y, ref_z))
            # But this double counts. The correct formula:
            # HV = sum_i (x_{i+1} - x_i) * H2({(y_j, z_j) : j >= i}, (ref_y, ref_z)) + (ref_x - x_{n-1}) * H2({(y_{n-1}, z_{n-1})}, (ref_y, ref_z))
            # where x_{n} = ref_x
            
            # Let's implement 2D hypervolume first
            def hv_2d(points_2d, ref_2d):
                """2D hypervolume for points < ref_2d."""
                m = len(points_2d)
                if m == 0:
                    return 0.0
                if m == 1:
                    return float((ref_2d[0] - points_2d[0][0]) * (ref_2d[1] - points_2d[0][1]))
                # Sort by first coordinate
                pts2 = points_2d[np.argsort(points_2d[:, 0])]
                hv = 0.0
                for i in range(m):
                    p = pts2[i]
                    # Width in x: from p[0] to next point's x or ref
                    x_next = pts2[i+1][0] if i+1 < m else ref_2d[0]
                    # Height in y: from min y of points j>=i to ref
                    y_min = np.min(pts2[i:, 1])
                    hv += (x_next - p[0]) * (ref_2d[1] - y_min)
                return hv

            # Now 3D HV using the slice method
            # Add a virtual point at ref
            hv = 0.0
            for i in range(n):
                # x-slice from x_i to x_{i+1} (or ref_x)
                x_i = pts[i][0]
                x_next = pts[i+1][0] if i+1 < n else ref_point[0]
                # In this x-slice, the active points are those with x <= x_i (i.e., indices 0..i)
                # The 2D HV of their y,z projections with ref (ref_y, ref_z)
                yz = pts[:i+1, 1:3]
                hv += (x_next - x_i) * hv_2d(yz, ref_point[1:3])
            return hv

        # Better approach: maintain a Pareto set and use HV to guide search
        # Use a simple but effective approach:
        # 1. Initialize with random points
        # 2. Maintain Pareto front
        # 3. For each new point, decide which parent to mutate based on HV contribution
        # 4. Use LHS or Latin hypercube for initial sampling
        
        rng = np.random.RandomState(0)
        
        # Initial population: use Latin Hypercube Sampling for better coverage
        pop_size = min(30, budget)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (rng.random(pop_size) + np.arange(pop_size)) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Filter valid points
        valid_mask = np.array([is_better_than_ref(f) for f in F])
        X_valid = X[valid_mask]
        F_valid = F[valid_mask]
        
        # Maintain Pareto set
        def is_pareto(F_arr):
            """Return mask of Pareto-optimal points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=bool)
            pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not pareto[i]:
                    continue
                for j in range(n):
                    if i == j or not pareto[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        pareto[i] = False
                        break
            return pareto

        # Compute HV contribution of each Pareto point
        def hv_contribution(point, pareto_set, ref_point):
            """Compute the HV contribution of a single point given the Pareto set."""
            if len(pareto_set) == 0:
                return float(np.prod(ref_point - point))
            # HV with point - HV without point
            with_pt = np.vstack([pareto_set, [point]])
            # Filter valid
            mask_with = np.all(with_pt < ref_point, axis=1)
            with_pt = with_pt[mask_with]
            mask_without = np.all(pareto_set < ref_point, axis=1)
            without_pt = pareto_set[mask_without]
            
            hv_with = _compute_hv_3d(np.array(with_pt)[np.argsort(with_pt[:, 0])], ref_point) if len(with_pt) > 0 else 0.0
            hv_without = _compute_hv_3d(np.array(without_pt)[np.argsort(without_pt[:, 0])], ref_point) if len(without_pt) > 0 else 0.0
            return hv_with - hv_without

        # Main loop
        pareto_X = X_valid.copy()
        pareto_F = F_valid.copy()
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Get current Pareto set
            if len(pareto_F) > 0:
                p_mask = is_pareto(pareto_F)
                pareto_F_cur = pareto_F[p_mask]
                pareto_X_cur = pareto_X[p_mask]
            else:
                pareto_F_cur = pareto_F
                pareto_X_cur = pareto_X
            
            # Select parent based on HV contribution (greedy)
            if len(pareto_F_cur) > 0:
                # Compute HV contribution for each Pareto point
                contribs = np.zeros(len(pareto_F_cur))
                for i in range(len(pareto_F_cur)):
                    contribs[i] = hv_contribution(pareto_F_cur[i], pareto_F_cur, ref)
                
                # Select parent with highest contribution (or randomly among top few)
                # Use softmax-like selection
                contribs = np.clip(contribs, 1e-10, None)
                probs = contribs / contribs.sum()
                parent_idx = rng.choice(len(pareto_F_cur), p=probs)
                parent = pareto_X_cur[parent_idx]
            else:
                # No valid points yet, pick random
                parent = X[rng.randint(len(X))]
            
            # Mutate: adaptive step size
            # Use a step size that decreases over time
            progress = used / budget
            step_scale = 0.5 * (1 - progress) + 0.05
            
            # Gaussian mutation
            child = parent + step_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archives
            if is_better_than_ref(fc):
                pareto_X = np.vstack([pareto_X, [child]])
                pareto_F = np.vstack([pareto_F, [fc]])
                X = np.vstack([X, [child]])
                F = np.vstack([F, [fc]])
            
            # Prune Pareto set to keep it manageable
            if len(pareto_F) > 100:
                p_mask = is_pareto(pareto_F)
                pareto_X = pareto_X[p_mask]
                pareto_F = pareto_F[p_mask]
                if len(pareto_F) > 100:
                    # Keep top 100 by HV contribution
                    contribs = np.zeros(len(pareto_F))
                    for i in range(len(pareto_F)):
                        contribs[i] = hv_contribution(pareto_F[i], pareto_F, ref)
                    top_idx = np.argsort(contribs)[-100:]
                    pareto_X = pareto_X[top_idx]
                    pareto_F = pareto_F[top_idx]
        
        return X

# EVOLVE-BLOCK-END
