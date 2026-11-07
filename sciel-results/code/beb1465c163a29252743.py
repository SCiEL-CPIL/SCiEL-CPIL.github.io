"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution_3d(self, points, ref):
        """
        Calculate the hypervolume contribution of each point in a 3D set.
        points: (N, 3) array.
        ref: (3,) array of reference point.
        Returns: (N,) array of HV contributions.
        """
        n = len(points)
        if n == 0:
            return np.array([])
        
        # Filter strictly better than ref
        mask = np.all(points < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        
        contrib = np.zeros(n)
        
        # Exact method: for each point i, contribution = HV(all) - HV(all \ {i})
        # We need an efficient HV calculation.
        
        def _calc_hv_3d(pts_subset):
            if len(pts_subset) == 0:
                return 0.0
            # Filter strictly better than ref
            msk = np.all(pts_subset < ref, axis=1)
            p = pts_subset[msk]
            if len(p) == 0:
                return 0.0
            
            # Sort by f0
            p = p[np.argsort(p[:, 0])]
            n_p = len(p)
            
            # Sweep line algorithm for 3D HV
            # Sort by x (f0).
            # For each point i (in sorted order), the contribution to HV is:
            # (x_next - x_i) * HV_2D(y, z of points j where x_j >= x_i, but considering only those not dominated in y,z by points k < i? No.)
            
            # Correct 3D HV sweep:
            # Sort points by x ascending.
            # Maintain a data structure of points that are "active" (not dominated in y,z by any previously processed point? No, that's not right either).
            
            # Let's use the "remove and recompute" with a grid-based HV for small N.
            # N is at most 20.
            # Grid method:
            # Unique x, y, z coords from points + ref.
            # For each cell in the grid, check if it is dominated by any point.
            # A cell [x_i, x_{i+1}] x [y_j, y_{j+1}] x [z_k, z_{k+1}] is dominated if there exists a point p such that
            # p.x <= x_i, p.y <= y_j, p.z <= z_k.
            
            xs = np.unique(np.concatenate([p[:, 0], [ref[0]]]))
            ys = np.unique(np.concatenate([p[:, 1], [ref[1]]]))
            zs = np.unique(np.concatenate([p[:, 2], [ref[2]]]))
            
            # We only care about cells where x < ref[0], y < ref[1], z < ref[2]
            # Since p < ref, all p coords are < ref.
            
            # Efficient check:
            # For a cell (i, j, k) with x_range [xs[i], xs[i+1]], y_range [ys[j], ys[j+1]], z_range [zs[k], zs[k+1]]
            # It is covered if min(p.x) <= xs[i] AND min(p.y) <= ys[j] AND min(p.z) <= zs[k] for some p? No.
            # It is covered if there exists p such that p.x <= xs[i] AND p.y <= ys[j] AND p.z <= zs[k].
            
            # Let's precompute for each x index i, the set of points with p.x <= xs[i].
            # Actually, simpler:
            # Iterate x from right to left? No, left to right.
            # Let's just do the O(N^4) check but vectorized if possible, or just accept O(N^4) for N=20.
            # 20^4 = 160,000 operations.
            # We do this N times (for each removal) -> 20 * 160,000 = 3,200,000 operations per selection step.
            # 200 steps -> 640,000,000 operations.
            # This might be too slow in pure Python loops, but NumPy helps.
            
            # Let's try a smarter O(N^3) or O(N^2 log N) method.
            # Standard 3D HV:
            # Sort by x.
            # hv = 0
            # For i in range(n):
            #   x_width = min(ref[0], x[i+1]) - x[i]  (if i < n-1 else ref[0] - x[i])
            #   # Need 2D HV of points j >= i in (y,z) space, but restricted to y,z < ref
            #   # And we must account for dominance from points k < i?
            #   # No, the standard algorithm is:
            #   # Sort by x.
            #   # For i from 0 to n-1:
            #   #   The "new" volume added by point i is the volume of the box [x_i, x_{i+1}) x [y_i, ref_y) x [z_i, ref_z)
            #   #   MINUS the volume already covered by points j < i in that x-slice?
            #   #   Actually, the correct decomposition is:
            #   #   Sort by x ascending.
            #   #   For i from 0 to n-1:
            #   #      x_width = (x[i+1] if i < n-1 else ref[0]) - x[i]
            #   #      # Consider the 2D HV of the set of points {p_j | j >= i} in the (y,z) plane,
            #   #      # but only counting the part that is NOT dominated by any p_k with k < i?
            #   #      # No, that's not right.
            
            # Let's use the "Union of Boxes" approximation or just the grid method with NumPy vectorization.
            
            # Grid Method Vectorized:
            # Create arrays of cell centers or just use the unique coords.
            # n_x = len(xs), n_y = len(ys), n_z = len(zs)
            # Cell volumes:
            # dx = np.diff(xs)
            # dy = np.diff(ys)
            # dz = np.diff(zs)
            # total_hv = 0
            # For each cell (i, j, k):
            #   is_dominated = False
            #   for p in p:
            #      if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
            #         is_dominated = True
            #         break
            #   if is_dominated:
            #      total_hv += dx[i] * dy[j] * dz[k]
            
            # This is O(n_x * n_y * n_z * N) = O(N^4).
            # With N=20, N^4 = 160k.
            # We can vectorize the inner loop.
            # For a fixed (i, j), we check all k? Or all points?
            
            # Let's try:
            # Precompute for each x index i, a mask of points with p.x <= xs[i].
            # This is not straightforward because xs are unique values from points.
            
            # Alternative:
            # Sort points by x.
            # For each x interval [xs[i], xs[i+1]], the points that can dominate cells in this interval
            # are those with p.x <= xs[i].
            # Let S_i be the set of points with p.x <= xs[i].
            # Then for this x-interval, we need to compute the 2D HV of S_i in the (y,z) plane.
            # The contribution to total HV is dx[i] * HV_2D(S_i).
            
            # This is O(N * N * HV_2D) = O(N^3 * N) = O(N^4) if HV_2D is O(N^2)?
            # HV_2D for M points is O(M^2) or O(M log M).
            # So total is sum_{i} (N - i) * O((N-i)^2) ~ O(N^4).
            
            # Let's implement HV_2D efficiently.
            
            def _hv_2d(pts_2d, ref_2d):
                # pts_2d: (M, 2), ref_2d: (2,)
                if len(pts_2d) == 0:
                    return 0.0
                msk = np.all(pts_2d < ref_2d, axis=1)
                p = pts_2d[msk]
                if len(p) == 0:
                    return 0.0
                
                # Sort by y
                p = p[np.argsort(p[:, 0])]
                n_p = len(p)
                hv = 0.0
                # Sweep from left to right (low y to high y)
                # Maintain max z? No, we need the area of the union of boxes [y_i, ref_y] x [z_i, ref_z].
                # Sort by y ascending.
                # For i from 0 to n-1:
                #   y_width = (y[i+1] if i < n-1 else ref_y) - y[i]
                #   # The height of the union in z for the points j >= i?
                #   # No, the standard 2D HV sweep:
                #   # Sort by y.
                #   # Keep track of the "active" z-values?
                #   # Actually, for 2D, it's simpler:
                #   # Sort by y ascending.
                #   # For i from 0 to n-1:
                #   #   The contribution is (y_next - y_i) * (max_z_of_points_j_ge_i - min_z? No)
                #   #   It's (y_next - y_i) * (ref_z - min_z_of_points_j_ge_i)? No.
                
                # Let's use the grid method for 2D too, or the "remove and recompute" for 2D is O(N^2).
                # For 2D, HV = sum of (ref_y - y_i) * (ref_z - z_i) for non-dominated points? No.
                
                # Simple 2D HV:
                # Sort by y.
                # hv = 0
                # max_z = 0 (or -inf)
                # Actually, sweep from right to left (high y to low y)?
                # Sort by y descending.
                # p = p[np.argsort(-p[:, 0])]
                # max_z = 0
                # for i in range(n_p):
                #    if p[i, 1] > max_z:
                #       y_width = (p[i-1, 0] if i > 0 else ref_2d[0]) - p[i, 0]
                #       # Wait, if we sort descending, the next point to the right has higher y.
                #       # The width is p[i, 0] - (p[i+1, 0] if i < n-1 else 0)? No.
                
                # Let's sort ascending by y.
                p = p[np.argsort(p[:, 0])]
                n_p = len(p)
                hv = 0.0
                # We need to find the "skyline" in z.
                # For each y-interval, the height is the min z of the points that cover it? No, max z?
                # The box is [y, ref_y] x [z, ref_z].
                # The union of these boxes.
                # Sort by y ascending.
                # For i from 0 to n-1:
                #   y_next = p[i+1, 0] if i < n-1 else ref_2d[0]
                #   # The points that start at y <= p[i, 0] and end at y >= y_next?
                #   # Actually, the height of the union at y is the min z of all points with p.y <= y <= p.y_next?
                #   # No, the box extends from p.y to ref_y.
                #   # So at any y >= p.y, the point contributes height (ref_z - p.z).
                #   # The union height at y is min(ref_z, max_{j: p_j.y <= y} (ref_z - p_j.z))? No, it's the min z of the active points.
                #   # Active points are those with p_j.y <= y.
                #   # The height of the union is ref_z - min_{j active} p_j.z.
                
                # So:
                # Sort by y ascending.
                # min_z = inf
                # prev_y = 0
                # for i in range(n_p):
                #    min_z = min(min_z, p[i, 1])
                #    y_next = p[i+1, 0] if i < n-1 else ref_2d[0]
                #    # The height is constant between p[i, 0] and y_next?
                #    # No, min_z can only decrease.
                #    # The height (ref_z - min_z) is constant in [p[i, 0], y_next) if no new point is added?
                #    # But new points are added at p[i, 0].
                #    # So the height changes at each p[i, 0].
                #    # The interval [p[i, 0], p[i+1, 0]) has height ref_z - min_z_i, where min_z_i is min z of points j <= i.
                #    hv += (y_next - p[i, 0]) * (ref_2d[1] - min_z)
                #    min_z = min(min_z, p[i, 1]) # Wait, min_z should be updated BEFORE calculating the width?
                #    # At y = p[i, 0], point i becomes active. So min_z includes p[i, 1].
                #    # So update min_z first.
                
                min_z = np.inf
                for i in range(n_p):
                    min_z = min(min_z, p[i, 1])
                    y_next = p[i+1, 0] if i < n-1 else ref_2d[0]
                    width = y_next - p[i, 0]
                    height = ref_2d[1] - min_z
                    if width > 0 and height > 0:
                        hv += width * height
                return hv
            
            # Now back to 3D:
            # Sort p by x ascending.
            p = p[np.argsort(p[:, 0])]
            n_p = len(p)
            hv = 0.0
            # For each x-interval [p[i, 0], x_next):
            #   The active points are those j <= i.
            #   We need the 2D HV of the active points in (y, z) space.
            #   But wait, the active points are those with p.j.x <= p[i, 0].
            #   So for interval i, the set of active points is {p_0, ..., p_i}.
            #   We need HV_2D({p_0.y, p_0.z}, ..., {p_i.y, p_i.z}), ref_yz.
            
            # This is O(N^2 * HV_2D) = O(N^2 * N) = O(N^3).
            # N=20 -> 8000 ops. Very fast.
            
            for i in range(n_p):
                x_next = p[i+1, 0] if i < n-1 else ref[0]
                width_x = x_next - p[i, 0]
                if width_x <= 0:
                    continue
                
                # Get active points (0 to i)
                active_pts = p[:i+1, 1:3] # y, z
                hv_2d = _hv_2d(active_pts, ref[1:3])
                hv += width_x * hv_2d
            
            return hv
        
        # Calculate total HV
        total_hv = _calc_hv_3d(points)
        
        # Calculate contribution for each point
        for i in range(n):
            if not mask[i]:
                contrib[i] = 0.0
                continue
            # Remove point i
            mask_remove = mask.copy()
            mask_remove[i] = False
            hv_without_i = _calc_hv_3d(points[mask_remove])
            contrib[i] = total_hv - hv_without_i
            
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                            
            nd_idx = np.where(~dominated)[0]
            
            if len(nd_idx) == 0:
                parent_idx = rng.integers(pop_size)
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent from non-dominated
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                
                # Select victim
                dom_idx = np.where(dominated)[0]
                if len(dom_idx) > 0:
                    victim_idx = dom_idx[int(np.argmax(np.sum(F[dom_idx], axis=1)))]
                else:
                    # All non-dominated. Use HV contribution to select victim.
                    contribs = self._hv_contribution_3d(F, ref)
                    # Remove the point with the smallest contribution
                    victim_idx = int(np.argmin(contribs))
            
            # Generate child
            parent = X[parent_idx]
            # Adaptive mutation scale
            progress = used / self.budget
            mutation_scale = 0.1 * (1.0 - 0.5 * progress)
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            X[victim_idx] = child
            F[victim_idx] = fc
            
        return X
# EVOLVE-BLOCK-END
