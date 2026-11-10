import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3.0, 5.0, 7.0])
        self.archive_size = 20
        self.start_time = 0.0
        self.deadline = 300.0

    def _hypervolume(self, points):
        """
        Calculate the hypervolume of a set of points in 3D with a reference point.
        points: (N, 3) array.
        Only points strictly better than ref_point are considered.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference point
        valid = np.all(points < self.ref_point, axis=1)
        pts = points[valid]
        
        if len(pts) == 0:
            return 0.0
        
        # Sort points by first objective (ascending)
        pts = pts[np.argsort(pts[:, 0])]
        
        hv = 0.0
        # We use the inclusion-exclusion principle or a simpler decomposition for 3D.
        # A common efficient method for small N is to iterate and calculate the unique volume.
        # For N=20, a simpler O(N^2) or O(N^3) approach is fine.
        
        # Let's use a simple recursive decomposition or just sum the boxes and subtract overlaps.
        # Actually, for 3D, we can project to 2D.
        # HV = sum over i of (ref_x - x_i) * HV_2D(projection of points j with x_j > x_i)
        
        # Simpler: Use a library-like approach for 3D HV.
        # Since N is small, we can compute the union of boxes.
        # Box for point i is [p_i, ref].
        
        # Let's implement a standard 3D HV algorithm.
        # Sort by x.
        # For each point i, the volume it uniquely contributes is:
        # (ref_x - x_i) * (volume of the 2D region in y,z that is not covered by any point j with x_j > x_i)
        
        # To find the 2D volume not covered by "future" points (those with larger x, i.e., worse in x1),
        # we can iterate from right to left (worst x1 to best x1).
        
        # Let's reverse the sort order for easier "dominance" checking in the loop.
        # Points sorted by x1 descending (worst first).
        pts_desc = pts[np.argsort(-pts[:, 0])]
        
        hv = 0.0
        # We need to track the "skyline" in 2D (y,z) for points that have been processed (which are worse in x1).
        # Actually, the standard algorithm iterates from best x1 to worst x1, maintaining the "frontier" in y,z.
        
        # Let's stick to the O(N^2) contribution method for simplicity and correctness, as N is small.
        # Total HV of set A.
        
        def hv_3d(pts):
            if len(pts) == 0: return 0.0
            # Sort by x1 ascending
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            # We will compute the contribution of each point.
            # A point p contributes volume (ref - p) minus volume covered by other points.
            # This is tricky.
            
            # Alternative: Decompose the space into slabs.
            # Slab i is defined by x in [x_i, x_{i+1}].
            # In each slab, the cross-section in (y,z) is the union of rectangles [y_j, ref_y] x [z_j, ref_z] for all points j with x_j <= x_slab_start.
            
            x_coords = np.unique(pts[:, 0])
            # Add reference x as a boundary
            x_coords = np.append(x_coords, self.ref_point[0])
            
            total_hv = 0.0
            for i in range(len(x_coords) - 1):
                x_start = x_coords[i]
                x_end = x_coords[i+1]
                dx = x_end - x_start
                if dx <= 0: continue
                
                # Find points that dominate the start of this slab (x_j <= x_start)
                # Actually, any point with x_j <= x_start contributes to the cross section at x_start.
                # The cross section is the union of rectangles defined by (y_j, z_j) for all j where x_j <= x_start.
                mask = pts[:, 0] <= x_start + 1e-9
                active_pts = pts[mask]
                
                if len(active_pts) == 0:
                    continue
                
                # Calculate 2D hypervolume (area) of the union of rectangles [y_j, ref_y] x [z_j, ref_z]
                area = self._hv_2d(active_pts[:, 1], active_pts[:, 2])
                total_hv += dx * area
                
            return total_hv

        def _hv_2d(ys, zs):
            """Calculate area of union of rectangles [y, ref_y] x [z, ref_z]."""
            if len(ys) == 0: return 0.0
            # Sort by y ascending
            order = np.argsort(ys)
            ys = ys[order]
            zs = zs[order]
            
            area = 0.0
            # The union of these rectangles can be calculated by sweeping y.
            # Sort by y.
            # For each y_i, the rectangle extends from y_i to ref_y.
            # The height (in z) available at y_i is min(z_j) for all j with y_j <= y_i? No.
            # It's the union.
            
            # Efficient 2D HV:
            # Sort points by y.
            # Keep track of the minimum z seen so far? No, we want the union of [z_j, ref_z].
            # So for a given y, the z-range covered is [min(z_j for j with y_j <= y), ref_z].
            # Wait, if y_j <= y, the rectangle j covers [y_j, ref_y] x [z_j, ref_z].
            # So at coordinate y, the z-range is [min(z_j | y_j <= y), ref_z].
            
            min_z = self.ref_point[2]
            prev_y = ys[0]
            # We need to integrate the length of the z-interval over y.
            # The interval changes only at y-coordinates of the points.
            
            # Let's iterate through unique y coordinates.
            y_coords = np.unique(ys)
            y_coords = np.append(y_coords, self.ref_point[1])
            
            # For each slab [y_i, y_{i+1}], the z-interval is determined by points with y_j <= y_i.
            # Let's precompute the min_z for each y_i.
            
            # Actually, simpler:
            # Sort by y ascending.
            # current_min_z = ref_z
            # area = 0
            # for i in range(len(ys)):
            #    # The segment from ys[i-1] to ys[i] has z-interval [current_min_z, ref_z]
            #    # But we need to update current_min_z with z[i] BEFORE adding the area for the segment starting at ys[i]?
            #    # No.
            #    # At y = ys[i], the new point i becomes active.
            #    # The segment [prev_y, ys[i]] was covered by points with y_j <= prev_y.
            #    # So we add area for [prev_y, ys[i]] using the min_z *before* updating with z[i].
            
            prev_y = ys[0]
            current_min_z = self.ref_point[2]
            
            # We need to handle the start. The first point ys[0] becomes active at y=ys[0].
            # So from y=ys[0] to the next y, the min_z is updated.
            
            # Let's do it step by step.
            # 1. Sort by y.
            # 2. Iterate.
            
            # Initialize
            area = 0.0
            current_min_z = self.ref_point[2]
            prev_y = ys[0]
            
            # We need to consider the segment from the first y to the second y, etc.
            # But what about y < ys[0]? No points there, so area 0.
            
            # Update min_z with the first point?
            # The point with smallest y is active for all y >= ys[0].
            
            # Let's use a sweep line.
            # Events: (y, z)
            # We want to integrate (ref_z - min_z_active(y)) dy.
            
            # Since N is small, we can just compute the union of rectangles directly.
            # Or use the property: Area = sum over i of (ref_y - y_i) * (ref_z - z_i) - overlaps.
            
            # Let's use the O(N^2) contribution method for 2D as well, or just a simple union algorithm.
            # Simple Union Algorithm for 2D:
            # 1. Sort points by y.
            # 2. Maintain a list of "active" z-intervals?
            
            # Actually, for N=20, we can just compute the area by checking each point's contribution.
            # Contribution of point i: (ref_y - y_i) * (ref_z - z_i) - area covered by other points j that overlap with i.
            # This is complex.
            
            # Let's use the slab method for 2D.
            # y_coords = unique ys + ref_y
            # For each slab [y_k, y_{k+1}]:
            #   active points: those with y_j <= y_k
            #   min_z = min(z_j for active points)
            #   if no active points, min_z = ref_z (area 0)
            #   height = ref_z - min_z
            #   width = y_{k+1} - y_k
            #   area += width * height
            
            y_unique = np.unique(ys)
            y_unique = np.append(y_unique, self.ref_point[1])
            
            area = 0.0
            for k in range(len(y_unique) - 1):
                y_start = y_unique[k]
                y_end = y_unique[k+1]
                width = y_end - y_start
                if width <= 0: continue
                
                # Find active points: y_j <= y_start
                active_mask = ys <= y_start + 1e-9
                if np.any(active_mask):
                    active_zs = zs[active_mask]
                    min_z = np.min(active_zs)
                    height = self.ref_point[2] - min_z
                    if height > 0:
                        area += width * height
                        
            return area

        return hv_3d(pts)

    def _hv_contribution(self, point, archive):
        """
        Calculate the hypervolume contribution of a single point to the archive.
        Contribution = HV(archive U {point}) - HV(archive)
        """
        hv_with = self._hypervolume(np.vstack([archive, point]))
        hv_without = self._hypervolume(archive)
        return hv_with - hv_without

    def _select_parent(self, archive, rng):
        """
        Select a parent from the archive using tournament selection based on HV contribution.
        """
        if len(archive) == 0:
            return None
        
        # Calculate contributions for all points in archive
        # This is expensive, so we might cache or just pick randomly if archive is small.
        # For N=20, calculating 20 contributions is 20 * (2 HV calcs) = 40 HV calcs.
        # Each HV calc is O(N^2) or O(N^3).
        # 200 evals * 40 HV calcs * 20^2 ops = 3.2M ops. Very fast.
        
        contributions = []
        for p in archive:
            c = self._hv_contribution(p, archive)
            contributions.append(c)
        
        contributions = np.array(contributions)
        
        # Tournament selection: pick 2, choose the one with higher contribution
        i1 = rng.integers(len(archive))
        i2 = rng.integers(len(archive))
        
        if contributions[i1] > contributions[i2]:
            return archive[i1]
        else:
            return archive[i2]

    def _crossover_mutation(self, p1, p2, lo, hi, rng):
        """
        SBX Crossover and Polynomial Mutation.
        """
        # Crossover
        # Use SBX with eta=15
        # Simple approximation: blend
        # x = p1 + 0.1 * (p2 - p1) ?
        # Let's use a simple blend with noise.
        
        # Actually, for small dim, a simple interpolation is good.
        # alpha ~ U(0, 1)
        alpha = rng.random(self.dim)
        child = alpha * p1 + (1 - alpha) * p2
        
        # Mutation
        # Gaussian mutation
        # Scale by range
        range_size = hi - lo
        child = child + 0.1 * range_size * rng.standard_normal(self.dim)
        
        # Clip
        child = np.clip(child, lo, hi)
        
        return child

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Initialize Archive
        archive = []
        rng = np.random.default_rng(0)
        
        # Initial population
        init_size = min(10, self.budget)
        for i in range(init_size):
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            if np.all(f < self.ref_point):
                archive.append(f)
        
        # If no valid points, add random ones until we have some
        while len(archive) < 5 and len(archive) < self.budget:
             x = lo + (hi - lo) * rng.random(dim)
             f = problem(x)
             if np.all(f < self.ref_point):
                 archive.append(f)
        
        if len(archive) == 0:
             # Fallback: just return random points
             return np.array([lo + (hi - lo) * rng.random(dim) for _ in range(10)])

        # Main Loop
        evals_used = len(archive)
        
        while evals_used < self.budget:
            # Deadline check
            if time.time() - self.start_time > self.deadline:
                break
            
            # Select parents
            p1 = self._select_parent(np.array(archive), rng)
            p2 = self._select_parent(np.array(archive), rng)
            
            if p1 is None or p2 is None:
                # If archive is empty or selection fails, pick random
                p1 = archive[rng.integers(len(archive))]
                p2 = archive[rng.integers(len(archive))]
            
            # Create child
            # Note: p1, p2 are objective vectors. We need the decision variables.
            # My current design stores only objective vectors in `archive`.
            # I need to store decision variables too to perform crossover.
            
            # REFACTOR: Store (x, f) pairs.
            
            # Let's restart the logic with (x, f) storage.
            pass 

    # The previous logic was flawed because I lost the decision variables.
    # Let's rewrite __call__ properly.
    
    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        archive = [] # List of (x, f)
        rng = np.random.default_rng(0)
        
        # Initial population
        init_size = min(10, self.budget)
        for i in range(init_size):
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            archive.append((x, f))
        
        evals_used = len(archive)
        
        while evals_used < self.budget:
            if time.time() - self.start_time > self.deadline:
                break
            
            # Convert archive to numpy arrays for easier manipulation
            X = np.array([a[0] for a in archive])
            F = np.array([a[1] for a in archive])
            
            # Filter valid points for HV calculation? 
            # The problem statement says "Only points strictly better than the reference point in every objective count."
            # But for selection, we should probably consider all points, or just valid ones?
            # If a point is worse than ref, its HV contribution is 0.
            # So we can just use all points.
            
            # Select parents based on HV contribution
            # To speed up, we can precompute contributions
            # But N is small.
            
            # Calculate contributions
            contributions = []
            for i in range(len(archive)):
                # Contribution of point i
                # HV(all) - HV(all except i)
                # This is O(N^2) HV calcs.
                # Let's just do it.
                
                # Create list of points excluding i
                F_without_i = np.delete(F, i, axis=0)
                
                hv_with = self._hypervolume(F)
                hv_without = self._hypervolume(F_without_i)
                contributions.append(hv_with - hv_without)
            
            contributions = np.array(contributions)
            
            # Tournament selection
            i1 = rng.integers(len(archive))
            i2 = rng.integers(len(archive))
            
            if contributions[i1] > contributions[i2]:
                p1_idx = i1
            else:
                p1_idx = i2
            
            i1 = rng.integers(len(archive))
            i2 = rng.integers(len(archive))
            
            if contributions[i1] > contributions[i2]:
                p2_idx = i1
            else:
                p2_idx = i2
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # Crossover and Mutation
            alpha = rng.random(dim)
            child = alpha * p1 + (1 - alpha) * p2
            child = child + 0.1 * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            evals_used += 1
            
            # Add to archive
            archive.append((child, fc))
            
            # If archive size > max_size, remove worst
            if len(archive) > self.archive_size:
                # Find point with lowest HV contribution
                # Recalculate contributions for the larger set?
                # Or just check the new point?
                # SMS-EMOA usually removes the worst from the combined set.
                # Let's just remove the point with the lowest contribution in the current archive.
                
                X_all = np.array([a[0] for a in archive])
                F_all = np.array([a[1] for a in archive])
                
                min_contrib = float('inf')
                min_idx = 0
                for i in range(len(archive)):
                    F_without_i = np.delete(F_all, i, axis=0)
                    hv_with = self._hypervolume(F_all)
                    hv_without = self._hypervolume(F_without_i)
                    c = hv_with - hv_without
                    if c < min_contrib:
                        min_contrib = c
                        min_idx = i
                
                archive.pop(min_idx)
        
        # Return the decision variables of the final archive
        if not archive:
            return np.array([])
        
        return np.array([a[0] for a in archive])
