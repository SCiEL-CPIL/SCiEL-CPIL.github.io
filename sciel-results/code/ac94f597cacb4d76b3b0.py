"""Multi-objective optimizer on WFG4 with hypervolume-guided coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided coordinate descent with ternary search per dimension."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])

        used = 0
        points = []  # list of (x, f)
        archive = []  # list of f-vectors for HV computation

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            used += 1
            f = np.array(problem(x))
            return f

        def hypervolume(fpts, ref):
            """Exact 3D hypervolume using inclusion-exclusion on sorted points."""
            if len(fpts) == 0:
                return 0.0
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(fpts < ref, axis=1)
            fpts = fpts[mask]
            if len(fpts) == 0:
                return 0.0

            # Sort by first objective
            order = np.argsort(fpts[:, 0])
            fpts = fpts[order]

            # For 3D, use the standard O(n^2) algorithm
            n = len(fpts)
            hv = 0.0
            for i in range(n):
                # Width in dim 0
                w0 = (fpts[i+1, 0] if i + 1 < n else ref[0]) - fpts[i, 0]
                # Points that dominate in dims 1,2: fpts[j,1] < fpts[i,1] and fpts[j,2] < fpts[i,2] for j > i
                # Actually for HV contribution of point i (sorted by dim0), we need the area in dims 1,2
                # that is NOT covered by any point j with j > i (i.e., with larger dim0) that has
                # smaller values in dims 1 and 2.
                # The contribution is w0 * (area in dims 1,2 covered by point i but not by any later point)
                
                # Find the region in dims 1,2: [fpts[i,1], ref[1]) x [fpts[i,2], ref[2])
                # minus regions covered by points j > i where fpts[j,1] < ref[1] and fpts[j,2] < ref[2]
                # A point j > i covers [fpts[j,1], ref[1]) x [fpts[j,2], ref[2]) if fpts[j,1] < ref[1] and fpts[j,2] < ref[2]
                # But we only care about the part that overlaps with point i's region.
                
                # This is getting complex. Let me use a simpler O(n^2) approach for 3D.
                pass

            # Let me use a cleaner 3D HV algorithm
            # Sort by dim 0
            # For each point i, its contribution to HV is:
            #   (next_x - x_i) * area_2d(points j > i that are "below" point i in dims 1,2)
            # where area_2d is the area of the rectangle [x_i,1, ref[1]) x [x_i,2, ref[2]) 
            #   minus the union of rectangles from points j > i that are within this region.
            
            # Actually, let me just implement a proper 3D HV.
            # Standard approach: sort by first coordinate, then for each point compute the 2D HV of
            # the points that come after it (in sorted order) but only considering points that are
            # "relevant" (i.e., have coordinates less than ref in the other dims).
            
            # Let me use a different, cleaner formulation.
            # For 3 objectives, HV can be computed as:
            # Sort points by f0. For point i, its contribution is:
            #   (f_{i+1,0} - f_{i,0}) * HV_2D({(f_{j,1}, f_{j,2}) : j > i, f_{j,1} < ref[1], f_{j,2} < ref[2]}, (ref[1], ref[2]))
            # where HV_2D is the 2D hypervolume.
            
            def hv_2d(pts, ref2d):
                """2D hypervolume: pts is Nx2, ref2d is [r1, r2]."""
                if len(pts) == 0:
                    return 0.0
                # Sort by first coordinate
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                n = len(pts)
                area = 0.0
                for i in range(n):
                    if pts[i, 0] >= ref2d[0] or pts[i, 1] >= ref2d[1]:
                        continue
                    w = (pts[i+1, 0] if i + 1 < n else ref2d[0]) - pts[i, 0]
                    h = ref2d[1] - pts[i, 1]
                    area += w * h
                return area

            total_hv = 0.0
            for i in range(n):
                if fpts[i, 0] >= ref[0]:
                    break
                # Next x boundary
                next_x = fpts[i+1, 0] if i + 1 < n else ref[0]
                w0 = next_x - fpts[i, 0]
                if w0 <= 0:
                    continue
                # Collect 2D points from j > i
                sub_pts = []
                for j in range(i+1, n):
                    if fpts[j, 1] < ref[1] and fpts[j, 2] < ref[2]:
                        sub_pts.append(fpts[j, 1:3])
                if sub_pts:
                    sub_pts = np.array(sub_pts)
                    area2d = hv_2d(sub_pts, ref[1:3])
                else:
                    area2d = (ref[1] - fpts[i, 1]) * (ref[2] - fpts[i, 2]) if fpts[i, 1] < ref[1] and fpts[i, 2] < ref[2] else 0.0
                # But we also need to account for point i's own position in 2D
                # The 2D region for point i is [fpts[i,1], ref[1]) x [fpts[i,2], ref[2])
                # minus the union of 2D regions from points j > i.
                # hv_2d above computes the HV of the sub_pts with reference ref[1:3], which gives
                # the area covered by sub_pts. But we want the area NOT covered by sub_pts within
                # the rectangle defined by point i.
                
                # Actually, the standard 3D HV algorithm works as follows:
                # Sort by dim 0. For point i, the contribution is:
                #   (x_{i+1} - x_i) * A_i
                # where A_i is the 2D area in the (dim1, dim2) plane that is covered by point i
                # but not by any point j with j > i (in sorted order).
                # A_i = area of [f_i1, ref1) x [f_i2, ref2) \ union_{j>i} [f_j1, ref1) x [f_j2, ref2)
                # This equals the 2D HV of the set {f_i} ∪ {f_j : j > i} minus the 2D HV of {f_j : j > i}
                # ... no, that's not quite right either.
                
                # Let me think again. The 3D HV is the volume of the union of boxes.
                # Box i is [f_i0, ref0) x [f_i1, ref1) x [f_i2, ref2).
                # When sorted by f_i0, box i's "new" volume is the part where dim0 is in [f_i0, f_{i+1,0})
                # and (dim1, dim2) is in the region covered by box i but not by any box j with j > i.
                # The (dim1, dim2) region for box i is [f_i1, ref1) x [f_i2, ref2).
                # The (dim1, dim2) regions for boxes j > i are [f_j1, ref1) x [f_j2, ref2).
                # The new area is the 2D HV of {f_i, f_{i+1}, ..., f_{n-1}} in dims 1,2 minus
                # the 2D HV of {f_{i+1}, ..., f_{n-1}} in dims 1,2.
                # Wait, that's not right either. The 2D HV of a set of points with reference (r1, r2)
                # is the area of the union of rectangles [p1, r1) x [p2, r2).
                # So A_i = HV_2D({f_i1, f_i2} ∪ {f_j1, f_j2 : j > i}, (ref1, ref2)) - HV_2D({f_j1, f_j2 : j > i}, (ref1, ref2))
                
                # Let me compute this properly.
                all_pts_2d = [fpts[i, 1:3]]
                for j in range(i+1, n):
                    if fpts[j, 1] < ref[1] and fpts[j, 2] < ref[2]:
                        all_pts_2d.append(fpts[j, 1:3])
                sub_pts_2d = []
                for j in range(i+1, n):
                    if fpts[j, 1] < ref[1] and fpts[j, 2] < ref[2]:
                        sub_pts_2d.append(fpts[j, 1:3])
                
                if all_pts_2d:
                    hv_all = hv_2d(np.array(all_pts_2d), ref[1:3])
                else:
                    hv_all = 0.0
                if sub_pts_2d:
                    hv_sub = hv_2d(np.array(sub_pts_2d), ref[1:3])
                else:
                    hv_sub = 0.0
                area_i = hv_all - hv_sub
                total_hv += w0 * area_i
            
            return total_hv

        def current_hv():
            if not archive:
                return 0.0
            farr = np.array(archive)
            return hypervolume(farr, ref)

        # Initial population: diverse random points
        n_init = min(30, self.budget // 5)
        for _ in range(n_init):
            if not time_ok() or used >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = evaluate(x)
            points.append((x.copy(), f))
            archive.append(f)

        best_hv = current_hv()

        # Coordinate descent with ternary search
        # For each point in the archive, try to improve it by optimizing each dimension
        # via ternary search, using HV contribution as the fitness.
        
        # We'll maintain a working set of candidate points
        # Start with the current archive points
        candidates = list(points)
        
        iteration = 0
        while time_ok() and used < self.budget:
            iteration += 1
            # Pick a candidate to improve (cycle through them)
            idx = iteration % len(candidates)
            x0, f0 = candidates[idx]
            
            # Try to improve x0 via coordinate descent
            improved = False
            for d in range(self.dim):
                if not time_ok() or used >= self.budget:
                    break
                
                # Ternary search on dimension d
                # We want to find x_d that maximizes the HV contribution of the new point
                # when combined with the existing archive.
                # We'll do a few iterations of ternary search.
                
                # Current best in this dimension search
                best_x_d = x0[d]
                best_f = f0
                best_hv_contrib = -1.0
                
                # Try a few random perturbations to get a sense of the landscape
                # Then refine with ternary search
                a = lo[d]
                b = hi[d]
                
                # Do ternary search for ~8 iterations
                for _ in range(8):
                    if not time_ok() or used >= self.budget:
                        break
                    m1 = a + (b - a) / 3
                    m2 = b - (b - a) / 3
                    
                    x1 = x0.copy()
                    x1[d] = m1
                    f1 = evaluate(x1)
                    
                    x2 = x0.copy()
                    x2[d] = m2
                    f2 = evaluate(x2)
                    
                    # Compute HV contribution: HV(archive ∪ {f1}) - HV(archive)
                    # But this is expensive. Instead, use a proxy:
                    # Check if f1 is non-dominated by archive, and estimate its contribution.
                    # For speed, use a simple proxy: sum of objectives (lower is better)
                    # combined with a check for dominance.
                    
                    # Actually, let's use the actual HV but only when the point is potentially
                    # non-dominated.
                    def is_dominated(f, arch):
                        for af in arch:
                            if np.all(af <= f) and np.any(af < f):
                                return True
                        return False
                    
                    # Use a cheaper fitness: if non-dominated, use negative sum; else use large value
                    def fitness(f):
                        if is_dominated(f, archive):
                            return 1e6
                        return -np.sum(f)
                    
                    fit1 = fitness(f1)
                    fit2 = fitness(f2)
                    
                    if fit1 < fit2:
                        b = m2
                    else:
                        a = m1
                
                # Evaluate the midpoint as the final candidate
                x_new = x0.copy()
                x_new[d] = (a + b) / 2
                if used < self.budget and time_ok():
                    f_new = evaluate(x_new)
                    if not is_dominated(f_new, archive):
                        # Check if it improves HV
                        # For now, accept if it's non-dominated
                        x0 = x_new
                        f0 = f_new
                        improved = True
            
            if improved:
                candidates[idx] = (x0.copy(), f0)
                # Add to archive if non-dominated
                if not is_dominated(f0, archive):
                    # Remove dominated points from archive
                    new_archive = []
                    for af in archive:
                        if not (np.all(f0 <= af) and np.any(f0 < af)):
                            new_archive.append(af)
                    new_archive.append(f0)
                    archive = new_archive
                    # Update best_hv occasionally (expensive)
                    if iteration % 10 == 0:
                        best_hv = current_hv()
            else:
                # Try a random restart from this point
                if used < self.budget and time_ok():
                    x_rand = x0 + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    x_rand = np.clip(x_rand, lo, hi)
                    f_rand = evaluate(x_rand)
                    if not is_dominated(f_rand, archive):
                        archive.append(f_rand)
                        candidates.append((x_rand.copy(), f_rand))

        return points
# EVOLVE-BLOCK-END
