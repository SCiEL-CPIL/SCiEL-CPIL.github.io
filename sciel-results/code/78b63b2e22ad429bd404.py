import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        pop_size = 12
        # LHS Init
        X = self._lhs(pop_size, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Store population as list of (x, f) or two arrays
        # Using two arrays is faster for vectorized ops
        X_pop = X
        F_pop = F
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated
            # F_pop is (N, 3)
            # A point i is dominated if exists j s.t. F_j <= F_i and F_j != F_i
            # Vectorized domination check:
            diff = F_pop[:, np.newaxis, :] - F_pop[np.newaxis, :, :]
            # j dominates i if F_j <= F_i and F_j != F_i
            # diff[i, j, k] = F_i[k] - F_j[k]
            # F_j <= F_i => diff >= 0
            # F_j != F_i => any diff > 0
            dom_mask = np.all(diff >= 0, axis=2) & np.any(diff > 0, axis=2)
            # dom_mask[i, j] is True if j dominates i
            is_dominated = np.any(dom_mask, axis=1)
            
            non_dom_idx = np.where(~is_dominated)[0]
            
            if len(non_dom_idx) == 0:
                # Should not happen, but fallback
                non_dom_idx = np.arange(len(F_pop))
            
            # Select parent
            parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            parent_x = X_pop[parent_idx]
            parent_f = F_pop[parent_idx]
            
            # Mutate
            t_ratio = used / self.budget
            sigma = 0.5 * (1 - t_ratio) + 0.01
            noise = rng.standard_normal(self.dim) * sigma * (hi - lo)
            child_x = np.clip(parent_x + noise, lo, hi)
            child_f = problem(child_x)
            used += 1
            
            # Add to population
            X_new = np.vstack([X_pop, child_x])
            F_new = np.vstack([F_pop, child_f])
            
            if len(X_new) > pop_size:
                # Remove point with lowest HV contribution
                # Calculate HV of full set
                hv_full = self._calc_hv(F_new, ref)
                min_contrib = float('inf')
                min_idx = -1
                for i in range(len(X_new)):
                    # Create sub-set
                    mask = np.ones(len(X_new), dtype=bool)
                    mask[i] = False
                    F_sub = F_new[mask]
                    hv_sub = self._calc_hv(F_sub, ref)
                    contrib = hv_full - hv_sub
                    if contrib < min_contrib:
                        min_contrib = contrib
                        min_idx = i
                
                # Remove min_idx
                mask = np.ones(len(X_new), dtype=bool)
                mask[min_idx] = False
                X_pop = X_new[mask]
                F_pop = F_new[mask]
            else:
                X_pop = X_new
                F_pop = F_new
        
        return X_pop

    def _lhs(self, n, lo, hi, rng):
        # Latin Hypercube Sampling
        # n samples, dim dimensions
        samples = np.zeros((n, self.dim))
        for d in range(self.dim):
            # Divide range into n equal intervals
            interval = (hi[d] - lo[d]) / n
            # Pick a random point in each interval
            points = lo[d] + interval * (np.arange(n) + rng.random(n))
            # Shuffle
            rng.shuffle(points)
            samples[:, d] = points
        return samples

    def _calc_hv(self, F, ref):
        # F is (N, 3), ref is (3,)
        # Filter points strictly better than ref
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Grid method
        # Get unique coords
        # To be robust, add ref coords
        xs = np.unique(np.concatenate([F[:, 0], [ref[0]]]))
        ys = np.unique(np.concatenate([F[:, 1], [ref[1]]]))
        zs = np.unique(np.concatenate([F[:, 2], [ref[2]]]))
        
        # Create grid edges
        # We need the volume of the union of boxes [x_i, ref_x] x [y_i, ref_y] x [z_i, ref_z]
        # This is equivalent to the volume of the region in the grid defined by
        # x in [min(F_x), ref_x], etc.
        
        # Efficient way:
        # Sort points by x.
        # Sweep along x.
        # Maintain a 2D structure for y,z.
        
        # Given small N, let's just use the grid cell counting.
        # Number of cells: (len(xs)-1) * (len(ys)-1) * (len(zs)-1)
        # Max 13*13*13 = 2197 cells.
        
        # Create 3D array of bools
        # Grid dimensions
        nx, ny, nz = len(xs), len(ys), len(zs)
        
        # We can vectorize the coverage check.
        # For each cell (i, j, k), the cell is covered if there exists a point p
        # such that p_x <= x_edge_i AND p_y <= y_edge_j AND p_z <= z_edge_k
        # Wait, the box is [p_x, ref_x].
        # So a cell [x_i, x_{i+1}] x [y_j, y_{j+1}] x [z_k, z_{k+1}] is covered
        # if there is a point p such that:
        # p_x <= x_i (actually p_x <= x_{i+1} is not enough, the box starts at p_x)
        # The box covers x in [p_x, ref_x].
        # So the cell [x_i, x_{i+1}] is covered if the interval [x_i, x_{i+1}] intersects [p_x, ref_x].
        # Since all points are < ref, and we sorted coords,
        # The union of boxes is the set of points (x,y,z) such that
        # there exists p in F with p_x <= x, p_y <= y, p_z <= z.
        
        # So, for each cell defined by corner (x_i, y_j, z_k) (the lower-left-far corner? No, the cell is between edges),
        # Let's use the "max" logic.
        # A point (x, y, z) in the grid is covered if max_x_covered >= x, max_y_covered >= y, max_z_covered >= z?
        # No, it's "exists p".
        
        # Let's iterate over cells.
        # Cell (i, j, k) corresponds to volume [xs[i], xs[i+1]] x [ys[j], ys[j+1]] x [zs[k], zs[k+1]].
        # It is covered if there exists p in F such that:
        # F[:, 0] <= xs[i+1] (box extends to ref, so if p_x is small, it covers large x)
        # Actually, the box is [p_x, ref_x].
        # So x is covered if x >= p_x and x <= ref_x.
        # Since our grid is within [min_p, ref], x <= ref is always true.
        # So x is covered by p if x >= p_x.
        # So the cell is covered if there exists p such that:
        # xs[i] >= p_x AND ys[j] >= p_y AND zs[k] >= p_z.
        # (Using the lower-left corner of the cell for the "greater than" check is sufficient?
        # If the lower-left corner is covered, the whole cell is covered?
        # Yes, because the coverage region is "upper right" from the point p.
        # If (x0, y0, z0) is covered, then any (x>x0, y>y0, z>z0) is also covered by the same p?
        # Wait. If p covers x0, then p_x <= x0.
        # If x > x0, then p_x <= x. So x is covered.
        # So yes, if the "min" corner of the cell is covered, the whole cell is covered.
        
        # So we need to check for each cell (i, j, k) if there exists p such that:
        # F[:, 0] <= xs[i]
        # F[:, 1] <= ys[j]
        # F[:, 2] <= zs[k]
        
        # We can vectorize this.
        # Create meshgrid for cell corners
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
        # Shape (nx, ny, nz)
        
        # For each cell, check if any point satisfies the condition.
        # Condition: (F[:, 0] <= Xg) & (F[:, 1] <= Yg) & (F[:, 2] <= Zg)
        # We need to do this for all points F.
        # This is a bit heavy to broadcast F (N, 3) against Grid (nx, ny, nz).
        
        # Alternative:
        # Sort points.
        # Since N is small, we can just loop over points and mark cells?
        # No, loop over cells is better.
        
        # Let's try a different approach:
        # Calculate the hypervolume by summing the volume of each "box" that is uniquely covered?
        # No, union is hard.
        
        # Let's stick to the cell check.
        # N=12, nx=13, ny=13, nz=13.
        # Grid size 2197.
        # For each of the 2197 cells, we check 12 points.
        # 2197 * 12 = 26k ops.
        # This is very fast.
        
        covered = np.zeros((nx, ny, nz), dtype=bool)
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    x_c = xs[i]
                    y_c = ys[j]
                    z_c = zs[k]
                    # Check if any point p has p_x <= x_c, p_y <= y_c, p_z <= z_c
                    # mask = (F[:, 0] <= x_c) & (F[:, 1] <= y_c) & (F[:, 2] <= z_c)
                    # if np.any(mask): covered[i,j,k] = True
                    
                    # Vectorized per cell
                    if np.any((F[:, 0] <= x_c) & (F[:, 1] <= y_c) & (F[:, 2] <= z_c)):
                        covered[i, j, k] = True
        
        # Calculate volume
        # dx = np.diff(xs)
        # dy = np.diff(ys)
        # dz = np.diff(zs)
        # vol = np.sum(covered * dx[:, None, None] * dy[None, :, None] * dz[None, None, :])
        
        dx = np.diff(xs)
        dy = np.diff(ys)
        dz = np.diff(zs)
        
        # Broadcast dx, dy, dz
        vol = 0.0
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    if covered[i, j, k]:
                        vol += dx[i] * dy[j] * dz[k]
        return vol
