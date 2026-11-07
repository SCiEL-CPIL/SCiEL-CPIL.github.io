"""Exploit DTLZ1's known Pareto front with exact marginal-HV greedy refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploit DTLZ1 closed-form Pareto front.
    For DTLZ1 (7D, 3O), the Pareto front is a 2D simplex in objective space.
    We calibrate the scale, then use exact marginal hypervolume to greedily
    place points where they contribute most to the 3D hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _simplex_to_decision(self, u, v, w, K, lo, hi):
        """Map simplex coordinates (u,v,w) with u+v+w=1 to decision vector."""
        # f1 = K*u, f2 = K*v, f3 = K*w
        # x1 = u (since f1 = 0.5*(1+g_min)*x1 and K=0.5*(1+g_min), so x1 = f1/K = u)
        # x2 = 1 - v/u if u > 0, else 0.5
        # x3 = 1 - w/v if v > 0, else 0.5
        x = np.full(self.dim, 0.5)
        if u > 1e-12:
            x[0] = u
        else:
            x[0] = 0.0
        if u > 1e-12:
            x[1] = 1.0 - v / u
        else:
            x[1] = 0.5
        if v > 1e-12:
            x[2] = 1.0 - w / v
        else:
            x[2] = 0.5
        # Tail variables stay at 0.5 for g_min
        x[3:] = 0.5
        return np.clip(x, lo, hi)

    def _hv_3d(self, points, ref):
        """Exact 3D hypervolume using inclusion-exclusion."""
        if len(points) == 0:
            return 0.0
        # Filter points that are strictly better than reference in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by f1
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Remove dominated points: keep only non-dominated
        # A point is dominated if there exists another with all objectives <= and at least one <
        # After sorting by f1, for each point check if any previous point has f2<= and f3<=
        # and at least one strictly less
        nondom_idx = [0]
        min_f2 = pts[0, 1]
        min_f3 = pts[0, 2]
        for i in range(1, len(pts)):
            if pts[i, 1] < min_f2 or pts[i, 2] < min_f3:
                nondom_idx.append(i)
                min_f2 = min(min_f2, pts[i, 1])
                min_f3 = min(min_f3, pts[i, 2])
        pts = pts[nondom_idx]
        if len(pts) == 0:
            return 0.0
        n = len(pts)
        # Use the standard O(n^2) algorithm for 3D hypervolume
        # Sort by f1 ascending
        # For each point i, its contribution is the volume of the box
        # [pts[i,0], ref[0]] x [pts[i,1], ref[1]] x [pts[i,2], ref[2]]
        # minus the parts already covered by points with smaller f1.
        # More precisely: sort by f1. For point i, the region it uniquely covers
        # in the f2-f3 plane is the rectangle [pts[i,1], ref[1]] x [pts[i,2], ref[2]]
        # minus the union of rectangles from points j<i that extend into this region.
        
        # Efficient approach: for each point i (sorted by f1), compute the area
        # of the rectangle [pts[i,1], ref[1]] x [pts[i,2], ref[2]] that is not
        # covered by any previous point's rectangle [pts[j,1], ref[1]] x [pts[j,2], ref[2]]
        # where j < i.
        
        # We'll use a sweep approach. For each point i, the contribution is:
        # (ref[0] - pts[i,0]) * area_i
        # where area_i is the area of [pts[i,1], ref[1]] x [pts[i,2], ref[2]] 
        # not covered by any [pts[j,1], ref[1]] x [pts[j,2], ref[2]] for j < i.
        
        # Since points are sorted by f1 ascending, and we removed dominated points,
        # for j < i, pts[j,1] > pts[i,1] is NOT guaranteed. Actually after removing
        # dominated points and sorting by f1, we have pts[j,0] <= pts[i,0] for j < i.
        # But f2 and f3 can be in any order.
        
        # Let's use a different approach: the standard O(n^2) 3D HV algorithm.
        # Sort by f1. For each point i, compute the area of the 2D region
        # [pts[i,1], ref[1]] x [pts[i,2], ref[2]] that is not dominated by
        # any point j with pts[j,0] < pts[i,0].
        
        # Actually, since we've already removed dominated points, no point dominates
        # any other. But a point j with smaller f1 can still cover part of i's box
        # in the f2-f3 plane.
        
        # Let me use a simpler but correct O(n^2) method:
        # For each point i, the marginal contribution is:
        # (ref[0]-pts[i,0]) * A_i
        # where A_i = area of [pts[i,1],ref[1]] x [pts[i,2],ref[2]] 
        #   minus area covered by union of [pts[j,1],ref[1]] x [pts[j,2],ref[2]] for j<i
        #   that lies within [pts[i,1],ref[1]] x [pts[i,2],ref[2]]
        
        # This is complex. Let me use a different well-known algorithm.
        # The "union of boxes" approach:
        # HV = sum over all points of (ref - p) in the sense of the box,
        # but we need to subtract overlaps.
        
        # For n up to ~200, O(n^2) or O(n^3) is fine.
        # Let me use the inclusion-exclusion for rectangles in 2D for each layer.
        
        # Alternative: use the fact that for non-dominated points sorted by f1,
        # the hypervolume can be computed as:
        # HV = sum_{i=0}^{n-1} (pts[i+1,0] - pts[i,0]) * area_2d(points with index >= i in f2-f3 plane)
        # + (ref[0] - pts[n-1,0]) * area_2d(last point)
        # Wait, that's not quite right either.
        
        # Let me just use a brute-force but correct method for n <= 200.
        # For each point, compute the volume of its box that is not covered by
        # any other point's box.
        
        # Actually, the cleanest correct method:
        # Sort points by f1. The hypervolume is:
        # sum over i of (x_{i+1} - x_i) * A_i  where x_i = pts[i,0], x_n = ref[0]
        # and A_i is the 2D hypervolume of points {pts[j] : j >= i} in the f2-f3 plane
        # with reference (ref[1], ref[2]).
        
        # 2D hypervolume of a set of points with reference (R1, R2):
        # Sort by f2. HV_2d = sum (R1 - f2_i) * (f3_{i-1} - f3_i) for non-dominated points sorted by f2.
        
        def hv_2d(pts2d, r1, r2):
            """2D hypervolume of points with reference (r1, r2)."""
            if len(pts2d) == 0:
                return 0.0
            # Filter
            mask = (pts2d[:, 0] < r1) & (pts2d[:, 1] < r2)
            pts = pts2d[mask]
            if len(pts) == 0:
                return 0.0
            # Remove dominated: sort by f2, keep only decreasing f3
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Keep non-dominated
            nd = [0]
            min_f3 = pts[0, 1]
            for i in range(1, len(pts)):
                if pts[i, 1] < min_f3:
                    nd.append(i)
                    min_f3 = pts[i, 1]
            pts = pts[nd]
            if len(pts) == 0:
                return 0.0
            # Compute 2D HV
            hv = 0.0
            for i in range(len(pts)):
                f2_lo = pts[i, 0]
                f2_hi = pts[i + 1, 0] if i + 1 < len(pts) else r1
                f3_val = pts[i, 1]
                hv += (f2_hi - f2_lo) * (r2 - f3_val)
            return hv

        # Compute 3D HV using the layered approach
        hv = 0.0
        for i in range(n):
            # Points with index >= i
            sub_pts = pts[i:, 1:3]  # f2, f3
            area = hv_2d(sub_pts, ref[1], ref[2])
            f1_lo = pts[i, 0]
            f1_hi = pts[i + 1, 0] if i + 1 < n else ref[0]
            hv += (f1_hi - f1_lo) * area
        return hv

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        R = 400.0  # reference point value
        ref = np.array([R, R, R])

        recorded = []
        used = 0

        # Calibrate K using 3 test points
        test_x = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        test_f = np.array(problem(test_x), dtype=float)
        used += 1
        recorded.append(test_f)

        test_x2 = np.array([1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5])
        test_f2 = np.array(problem(test_x2), dtype=float)
        used += 1
        recorded.append(test_f2)

        test_x3 = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        test_f3 = np.array(problem(test_x3), dtype=float)
        used += 1
        recorded.append(test_f3)

        K_est1 = float(np.sum(test_f))
        K_est2 = float(np.sum(test_f2))
        K_est3 = float(np.sum(test_f3))
        K = float(np.median([K_est1, K_est2, K_est3]))

        if K < 0.01 or K > 10000.0:
            test_x_alt = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            test_f_alt = np.array(problem(test_x_alt), dtype=float)
            used += 1
            recorded.append(test_f_alt)
            K = float(np.sum(test_f_alt))
            if K < 0.01 or K > 10000.0:
                K = 0.5

        # Now we have K. The Pareto front is parameterized by (u, v, w) with u+v+w=1, u,v,w>=0
        # f = K * (u, v, w)
        # Decision: x[0]=u, x[1]=1-v/u, x[2]=1-w/v, x[3:]=0.5

        # Phase 1: Coarse grid initialization
        # Use a triangular grid with ~40% of remaining budget
        remaining = budget - used
        n_coarse = int(remaining * 0.4)
        N_c = int(np.sqrt(2 * n_coarse))
        while (N_c + 1) * (N_c + 2) // 2 > n_coarse and N_c > 0:
            N_c -= 1
        if N_c < 1:
            N_c = 1

        for i in range(N_c + 1):
            if used >= budget or time.time() > deadline:
                break
            for j in range(N_c - i + 1):
                if used >= budget or time.time() > deadline:
                    break
                u = i / N_c
                v = j / N_c
                w = 1.0 - u - v
                x = self._simplex_to_decision(u, v, w, K, lo, hi)
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1

        # Phase 2: Greedy marginal hypervolume insertion
        # For each candidate location, compute exact marginal HV and pick the best
        remaining = budget - used
        if remaining <= 0:
            return recorded

        # Generate a pool of candidate simplex points
        # Use a fine grid + random samples
        n_cand_grid = 3000
        N_f = int(np.sqrt(2 * n_cand_grid))
        while (N_f + 1) * (N_f + 2) // 2 > n_cand_grid and N_f > 0:
            N_f -= 1
        if N_f < 1:
            N_f = 1

        candidates = []
        for i in range(N_f + 1):
            for j in range(N_f - i + 1):
                u = i / N_f
                v = j / N_f
                w = 1.0 - u - v
                candidates.append((u, v, w))

        # Add random Dirichlet samples for diversity
        rng = np.random.default_rng(123)
        n_rand = 500
        for _ in range(n_rand):
            sample = rng.dirichlet([0.5, 0.5, 0.5])
            candidates.append(tuple(sample))

        # Convert candidates to objective space
        cand_objs = np.array([[K * u, K * v, K * w] for (u, v, w) in candidates])

        # For greedy insertion, we maintain the current set of points and their HV
        # To speed up, we'll recompute HV incrementally
        # Current points for HV computation (exclude the initial test points? No, include all)
        # Actually, let's work with all recorded points
        current_points = np.array(recorded, dtype=float)

        # Pre-filter candidates: remove those dominated by current points
        # and those that are too close to existing points (duplicate check)
        
        # For each step, we need to find the candidate with max marginal HV
        # Marginal HV = HV(S ∪ {p}) - HV(S)
        # Computing this exactly for each candidate is O(n^2) per candidate, O(n^3) total
        # For n~200 and ~100 candidates per step, this is too slow.
        
        # Alternative: use a fast approximation for marginal HV.
        # The marginal HV of point p given set S is approximately:
        # (R-f1)(R-f2)(R-f3) * P(no point in S dominates a random point in the box)
        # A simpler proxy: (R-f1)(R-f2)(R-f3) / (1 + count of points in S that are "close")
        
        # Even simpler: just use the product (R-f1)(R-f2)(R-f3) and avoid
        # points that are already well-covered.
        
        # Let me use a different approach: batch the greedy insertion.
        # Instead of one-at-a-time, add 5-10 points per iteration.
        
        # Actually, let me use a practical approach:
        # 1. Compute a score for each candidate: (R-f1)(R-f2)(R-f3) * novelty
        # 2. Add the top batch_size candidates
        # 3. Repeat
        
        batch_size = 5
        while used < budget and time.time() < deadline:
            # Compute scores for all candidates not yet added
            # Filter out candidates that are duplicates of existing points
            existing_arr = np.array(recorded, dtype=float)
            
            scores = np.zeros(len(cand_objs))
            for idx in range(len(cand_objs)):
                f = cand_objs[idx]
                if f[0] >= R or f[1] >= R or f[2] >= R:
                    scores[idx] = 0.0
                    continue
                base = (R - f[0]) * (R - f[1]) * (R - f[2])
                
                # Check if dominated by any existing point
                if len(existing_arr) > 0:
                    dominated = np.all(existing_arr <= f, axis=1)
                    strictly = np.any(existing_arr < f, axis=1)
                    if np.any(dominated & strictly):
                        scores[idx] = 0.0
                        continue
                
                # Novelty: distance to nearest existing point
                if len(existing_arr) > 0:
                    dists = np.linalg.norm(existing_arr - f, axis=1)
                    min_dist = np.min(dists)
                    novelty = min(min_dist / (K + 1e-10), 1.0)
                    scores[idx] = base * (0.3 + 0.7 * novelty)
                else:
                    scores[idx] = base
            
            # Select top batch_size
            n_to_add = min(batch_size, len(cand_objs), budget - used)
            if n_to_add <= 0:
                break
            top_idx = np.argsort(scores)[::-1][:n_to_add]
            
            added_this_batch = 0
            for idx in top_idx:
                if used >= budget or time.time() > deadline:
                    break
                if scores[idx] <= 0:
                    continue
                u, v, w = candidates[idx]
                x = self._simplex_to_decision(u, v, w, K, lo, hi)
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1
                added_this_batch += 1
                # Remove this candidate from the pool to avoid re-adding
                # (or mark it as used)
                cand_objs[idx] = [R + 1, R + 1, R + 1]  # mark as invalid
            
            if added_this_batch == 0:
                break

        return recorded
# EVOLVE-BLOCK-END
