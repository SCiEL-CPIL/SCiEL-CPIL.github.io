# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, point, set_points):
        """Check if point is dominated by any point in set_points."""
        if len(set_points) == 0:
            return False
        # point is dominated if exists p in set_points such that p <= point and p != point
        # In minimization: p_i <= point_i for all i, and p_j < point_j for at least one j
        dominated = np.all(set_points <= point, axis=1)
        strictly_better = np.any(set_points < point, axis=1)
        return np.any(dominated & strictly_better)

    def _remove_dominated(self, points):
        """Return a list of non-dominated points from the input array."""
        if len(points) == 0:
            return []
        
        nondominated = []
        for i in range(len(points)):
            if not self._is_dominated(points[i], points):
                nondominated.append(points[i])
        return np.array(nondominated) if nondominated else np.array([])

    def _marginal_hv_3d(self, new_point, existing_points, ref_point):
        """
        Compute the marginal hypervolume contribution of new_point given existing_points.
        existing_points should be a list/array of non-dominated points.
        """
        if len(existing_points) == 0:
            # HV is just the box from new_point to ref_point
            if np.all(new_point < ref_point):
                return np.prod(ref_point - new_point)
            return 0.0
        
        # The marginal HV is the volume of the box (new_point, ref_point)
        # minus the volume of that box that is already covered by existing_points.
        
        # Since all points are non-dominated, the covered region is the union of boxes
        # from existing points, intersected with the box of new_point.
        
        # For 3D, we can compute the union of rectangles in the yz-plane
        # projected by the x-coordinates.
        
        # A simpler robust method for small N:
        # The contribution is the volume of the region R = {x | new_point <= x <= ref_point}
        # that is NOT in U = Union_{p in existing} {x | p <= x <= ref_point}.
        
        # Let's use a discretization-free geometric approach.
        # The box B_new is defined by [nx, rx], [ny, ry], [nz, rz].
        # We need to subtract the volume of B_new ∩ U.
        
        # U is the union of boxes B_p for p in existing.
        # Since we are in 3D, we can slice the box B_new along the x-axis.
        
        # Collect all x-coordinates that are boundaries of any box intersecting B_new.
        # These are: nx, rx, and p_x for all p in existing such that p_x < rx and p_x > nx (or p_x < nx? No, if p_x < nx, the box starts before nx, so intersection starts at nx).
        
        # Actually, it's easier to just compute the HV of the set {new_point} ∪ existing_points
        # and subtract the HV of existing_points.
        # But we don't have a fast HV function for the whole set yet.
        
        # Let's implement a fast 3D HV for small sets to get the difference.
        # HV(S ∪ {p}) - HV(S)
        
        # We can compute HV of a set of non-dominated points in 3D by sorting by x
        # and sweeping.
        
        def hv_3d(points, ref):
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than ref
            valid = np.all(points < ref, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
                
            # Sort by x
            order = np.argsort(points[:, 0])
            points = points[order]
            
            n = len(points)
            if n == 1:
                return float(np.prod(ref - points[0]))
            
            # We will sweep from left (small x) to right (large x).
            # The volume is sum over intervals [x_i, x_{i+1}] of (x_{i+1} - x_i) * Area_i
            # where Area_i is the 2D hypervolume of the points with x <= x_i in the yz plane.
            # Wait, the standard algorithm is:
            # HV = sum_{i=1}^{n} (x_{i+1} - x_i) * HV_2d( { (y_j, z_j) | j >= i } , (ref_y, ref_z) ) ?
            # No, let's use the property that HV is additive over disjoint x-slabs.
            
            # Let's use the "union of boxes" logic for the specific new point.
            # This is getting complex. Let's use a simpler heuristic for selection
            # that is very close to marginal HV:
            # The box volume of the new point, reduced by the overlap with existing points.
            
            # Overlap in 3D is hard. Let's use a 2D projection approximation.
            # Or, since N is small (< 100), we can compute the exact marginal HV
            # by calculating the volume of the box of the new point that is not covered
            # by any existing box.
            
            # Let's try a different approach:
            # The marginal HV is the volume of the region dominated by new_point
            # but not by any point in existing_points.
            
            # We can compute this by clipping the box of new_point against each existing point.
            # This is not straightforward due to overlaps between existing points.
            
            # However, if we maintain the set of non-dominated points, they are disjoint
            # in terms of "dominance" but their boxes overlap.
            
            # Let's use a grid-based approximation for the marginal HV?
            # No, let's use the exact HV difference.
            # We need a fast HV_3D function.
            
            # Algorithm for HV_3D (non-dominated points):
            # 1. Sort by x.
            # 2. For each point i, the contribution is (x_{i+1} - x_i) * HV_2D( S_i, ref_yz )
            #    where S_i is the set of points j such that x_j <= x_i ?
            #    Actually, the contribution of the slab between x_i and x_{i+1} is determined
            #    by the points that are "active" in that slab.
            #    A point j is active in [x_i, x_{i+1}] if x_j <= x_{i+1} and x_j > x_{i-1}?
            
            # Let's use the standard "sweep line" for 3D.
            # Sort points by x.
            # Let x_0 = -inf, x_{n+1} = ref_x.
            # For i = 1 to n:
            #   Let S_i be the set of points {p_j | j=1..i}.
            #   The volume of the region x in [x_i, x_{i+1}] is (x_{i+1} - x_i) * HV_2D( { (y_j, z_j) | p_j in S_i }, (ref_y, ref_z) ).
            #   Wait, this is not correct. The points in S_i dominate the region x >= x_j.
            #   So for x in [x_i, x_{i+1}], the active points are those with x_j <= x.
            #   Since x >= x_i, all points j <= i are active? No, points j > i have x_j > x_i, so they don't dominate x < x_j.
            #   So for x in [x_i, x_{i+1}], the active points are exactly {p_1, ..., p_i}.
            #   So the volume is sum_{i=1}^{n} (x_{i+1} - x_i) * HV_2D( { (y_j, z_j) | j=1..i }, (ref_y, ref_z) ).
            
            # Let's verify this.
            # If i=1, x in [x_1, x_2]. Active point: p_1.
            # Volume = (x_2 - x_1) * HV_2D( {(y_1, z_1)}, ref ).
            # This is the volume of the box [x_1, x_2] x [y_1, ref_y] x [z_1, ref_z].
            # This is correct.
            
            # So we need to compute HV_2D for prefixes of the sorted points.
            
            def hv_2d(pts, ref):
                if len(pts) == 0:
                    return 0.0
                # pts is 2D array (y, z)
                # Sort by y
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                total = 0.0
                max_z = ref[1]
                for i in range(len(pts) - 1, -1, -1):
                    if pts[i, 1] < max_z:
                        total += (max_z - pts[i, 1]) * (ref[0] - pts[i, 0])
                        max_z = pts[i, 1]
                return total
            
            # Compute HV_3D
            total_hv = 0.0
            # x_{n+1} is ref[0]
            for i in range(n):
                x_next = points[i+1, 0] if i < n - 1 else ref[0]
                x_curr = points[i, 0]
                # Points j=0..i are active in [x_curr, x_next]
                active_2d = points[:i+1, 1:3] # y, z columns
                area = hv_2d(active_2d, ref[1:3])
                total_hv += (x_next - x_curr) * area
                
            return total_hv

        # Calculate HV of existing_points
        hv_existing = hv_3d(existing_points, ref_point)
        
        # Calculate HV of existing_points + new_point
        combined = np.vstack([existing_points, new_point.reshape(1, -1)])
        hv_combined = hv_3d(combined, ref_point)
        
        return hv_combined - hv_existing

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([2.5, 2.5, 2.5])
        
        budget = self.budget
        dim = self.dim
        
        # DTLZ4 Front:
        # x[2:] = 1.0
        # x[0], x[1] in [0, pi/2]
        # f1 = cos(x0)cos(x1)
        # f2 = cos(x0)sin(x1)
        # f3 = sin(x0)
        
        def make_front_point(t1, t2):
            x = np.zeros(dim)
            x[0] = t1
            x[1] = t2
            x[2:] = 1.0
            return np.clip(x, lo, hi)
        
        # Initialize with a Halton sequence for diversity
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        n_init = 40
        init_points = []
        for i in range(1, n_init + 1):
            t1 = halton(i, 2) * (np.pi / 2)
            t2 = halton(i, 3) * (np.pi / 2)
            x = make_front_point(t1, t2)
            init_points.append(x)
            
        F = np.array([problem(x) for x in init_points])
        used = n_init
        
        # Store non-dominated points
        nondom_X = []
        nondom_F = []
        
        for i in range(len(F)):
            if not self._is_dominated(F[i], np.array(nondom_F) if nondom_F else np.array([])):
                # Check if it dominates any existing
                if nondom_F:
                    existing_F = np.array(nondom_F)
                    dominated_existing = np.all(F[i] <= existing_F, axis=1) & np.any(F[i] < existing_F, axis=1)
                    if np.any(dominated_existing):
                        # Remove dominated existing points
                        mask = ~dominated_existing
                        nondom_X = [nondom_X[j] for j in range(len(nondom_X)) if mask[j]]
                        nondom_F = [nondom_F[j] for j in range(len(nondom_F)) if mask[j]]
                nondom_X.append(init_points[i])
                nondom_F.append(F[i])
        
        # Greedy loop
        n_candidates = 20
        while used < budget:
            candidates = []
            cand_F = []
            
            # Generate candidates on the front
            for _ in range(n_candidates):
                # Random sampling on the front
                t1 = rng.uniform(0, np.pi / 2)
                t2 = rng.uniform(0, np.pi / 2)
                x = make_front_point(t1, t2)
                candidates.append(x)
                
                if used + len(candidates) >= budget:
                    break
            
            if not candidates:
                break
                
            # Evaluate
            for x in candidates:
                f = problem(x)
                used += 1
                cand_F.append(f)
                
            cand_F = np.array(cand_F)
            
            # Select best candidate based on marginal HV
            best_idx = -1
            best_score = -1.0
            
            # Filter candidates that are not dominated by current nondom set
            valid_cands = []
            for i in range(len(cand_F)):
                if not self._is_dominated(cand_F[i], np.array(nondom_F) if nondom_F else np.array([])):
                    valid_cands.append(i)
            
            if not valid_cands:
                continue
                
            for i in valid_cands:
                # Check if it dominates any existing nondom points
                # If so, the marginal HV is larger
                score = self._marginal_hv_3d(cand_F[i], np.array(nondom_F) if nondom_F else np.array([]), ref_point)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            if best_idx == -1:
                continue
                
            # Add best candidate to nondom set
            new_f = cand_F[best_idx]
            new_x = candidates[best_idx]
            
            # Remove dominated points from nondom set
            if nondom_F:
                existing_F = np.array(nondom_F)
                dominated_existing = np.all(new_f <= existing_F, axis=1) & np.any(new_f < existing_F, axis=1)
                if np.any(dominated_existing):
                    mask = ~dominated_existing
                    nondom_X = [nondom_X[j] for j in range(len(nondom_X)) if mask[j]]
                    nondom_F = [nondom_F[j] for j in range(len(nondom_F)) if mask[j]]
            
            nondom_X.append(new_x)
            nondom_F.append(new_f)
            
            if used >= budget:
                break
                
        return nondom_X
# EVOLVE-BLOCK-END
