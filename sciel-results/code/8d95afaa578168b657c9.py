"""Hypervolume-driven adaptive multi-objective optimizer for RE37."""
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive sampling: greedily fill gaps in objective space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """Compute exact 3D hypervolume for non-dominated points relative to ref."""
        n = len(points)
        if n == 0:
            return 0.0
        # Filter points strictly better than reference in all objectives
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # For 3D, we can use inclusion-exclusion or sweep
        # Simple approach: for each point, compute its contribution
        # A point contributes if it's not dominated by any other point
        # Contribution of point i = (ref[0]-f_i[0]) * area in 2D of region
        # where no other point dominates in dims 1,2
        
        # More efficient: sort and use incremental update
        # For small n, O(n^2) is fine
        hv = 0.0
        for i in range(len(pts)):
            f = pts[i]
            # Check if this point is dominated by any earlier point
            dominated = False
            for j in range(i):
                if np.all(pts[j] <= f) and np.any(pts[j] < f):
                    dominated = True
                    break
            if dominated:
                continue
            # Compute 2D area of the box from f[1:] to ref[1:] not covered by other points
            # that dominate in dim 0 (i.e., have f0 <= f[0])
            # Since sorted by dim 0, all j < i have f0 <= f[0]
            # We need the area in (dim1, dim2) space of the rectangle [f[1], ref[1]] x [f[2], ref[2]]
            # minus areas covered by points j < i that also have f1 <= f[1] and f2 <= f[2]
            
            # Actually, for exact hypervolume in 3D with sorted points, we can use:
            # The contribution is (ref[0]-f[0]) * A where A is the 2D hypervolume
            # of the set of points {pts[j][1:] for j < i if pts[j][1] < f[1] and pts[j][2] < f[2]}
            # plus the corner (f[1], f[2])
            
            # Simpler: just compute 2D hypervolume of all points j<=i projected to dims 1,2
            # but that's not quite right either.
            
            # Let me use a cleaner approach: compute 3D HV by iterating
            # For each point, its marginal contribution is the volume of the box
            # [f, ref] that is not covered by any other point's box.
            
            # For n up to ~50, let's do O(n^3) which is fine
            # Actually let me use a standard algorithm
            
        # Use a standard 3D hypervolume algorithm
        # Sort by dim 0, then for each point compute 2D HV of remaining points' projections
        # that are "active" (not dominated in dims 1,2 by the current point's predecessors)
        
        # Simpler correct approach for small n:
        # Hypervolume = sum over all non-dominated points of their "exclusive" volume
        # A point p is non-dominated if no other point q has q <= p and q != p
        
        # For 3D, we can use:
        # Sort by dim0 ascending
        # Maintain a list of "active" points in the 2D space (dims 1,2)
        # When adding a new point, compute the 2D area it exclusively covers
        
        # Let me just use a straightforward O(n^2) method for 3D:
        # The hypervolume is the union of boxes [f_i, ref] for all i.
        # For 3D with small n, we can compute it by:
        # 1. Get all unique coordinates in each dimension (including ref)
        # 2. Create a 3D grid
        # 3. For each cell, check if it's covered by any point's box
        # This is O(n^3 * 8) which is fine for n < 100
        
        # Even simpler for our case: since we only have ~50-100 points max,
        # let's use the incremental approach properly.
        
        # Standard 3D HV algorithm:
        # Sort points by dim 0
        # For each point in order, compute the 2D hypervolume of the set of points
        # that are "visible" from this point in the 2D projection, times the width in dim 0.
        
        # Actually, the cleanest correct way for small n:
        n_pts = len(pts)
        if n_pts == 0:
            return 0.0
        
        # Get all unique x-coordinates
        xs = sorted(set([pts[i, 0] for i in range(n_pts)] + [ref[0]]))
        ys = sorted(set([pts[i, 1] for i in range(n_pts)] + [ref[1]]))
        zs = sorted(set([pts[i, 2] for i in range(n_pts)] + [ref[2]]))
        
        # For each cell in the 3D grid, check if it's dominated by any point
        # A cell [x_i, x_{i+1}] x [y_j, y_{j+1}] x [z_k, z_{k+1}] is covered
        # if there exists a point p with p[0] <= x_i, p[1] <= y_j, p[2] <= z_k
        
        total = 0.0
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(zs) - 1):
                    x_lo, x_hi = xs[i], xs[i+1]
                    y_lo, y_hi = ys[j], ys[j+1]
                    z_lo, z_hi = zs[k], zs[k+1]
                    # Cell is covered if some point has all coords <= cell lower bounds
                    covered = False
                    for m in range(n_pts):
                        if pts[m, 0] <= x_lo and pts[m, 1] <= y_lo and pts[m, 2] <= z_lo:
                            covered = True
                            break
                    if covered:
                        total += (x_hi - x_lo) * (y_hi - y_lo) * (z_hi - z_lo)
        return total

    def _marginal_hv(self, candidate, points, ref):
        """Compute marginal hypervolume contribution of adding candidate to points."""
        # HV with candidate - HV without
        # But we only care about the increase, so compute HV of points + candidate
        # and subtract HV of points
        # For efficiency, since n is small, just compute both
        if len(points) == 0:
            return self._hypervolume_3d(np.array([candidate]), ref)
        old_hv = self._hypervolume_3d(points, ref)
        new_pts = np.vstack([points, candidate])
        new_hv = self._hypervolume_3d(new_pts, ref)
        return new_hv - old_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1.1, 1.3, 1.1])
        
        # Initialize with random points
        n_init = min(20, self.budget // 10)
        X = lo + (hi - lo) * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Keep track of all evaluated points and their objectives
        all_X = X.copy()
        all_F = F.copy()
        
        # Find initial non-dominated set
        def non_dominated_mask(F_arr):
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask
        
        # Adaptive sampling loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current non-dominated set
            nd_mask = non_dominated_mask(all_F)
            nd_X = all_X[nd_mask]
            nd_F = all_F[nd_mask]
            
            # Generate candidate proposals
            n_candidates = 50
            candidates = []
            
            # Strategy 1: Perturb non-dominated points
            for i in range(min(len(nd_X), 10)):
                for step in [0.05, 0.1, 0.2]:
                    c = nd_X[i] + step * (hi - lo) * rng.standard_normal(self.dim)
                    c = np.clip(c, lo, hi)
                    candidates.append(c)
            
            # Strategy 2: Random points
            for _ in range(20):
                c = lo + (hi - lo) * rng.random(self.dim)
                candidates.append(c)
            
            # Strategy 3: Interpolate between pairs of non-dominated points
            if len(nd_X) >= 2:
                for _ in range(10):
                    i, j = rng.integers(0, len(nd_X), size=2)
                    if i != j:
                        t = rng.random()
                        c = t * nd_X[i] + (1 - t) * nd_X[j]
                        c = np.clip(c, lo, hi)
                        candidates.append(c)
            
            # Evaluate candidates and pick the one with best marginal HV
            best_marginal = -np.inf
            best_candidate = None
            best_fc = None
            
            for c in candidates:
                if time.time() > deadline:
                    break
                fc = problem(c)
                used += 1
                all_X = np.vstack([all_X, c])
                all_F = np.vstack([all_F, fc])
                
                # Compute marginal HV
                if len(nd_F) > 0:
                    marginal = self._marginal_hv(fc, nd_F, ref)
                else:
                    marginal = self._hypervolume_3d(np.array([fc]), ref)
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_candidate = c
                    best_fc = fc
            
            # If we ran out of time during candidate eval, break
            if time.time() > deadline:
                break
        
        return all_X
