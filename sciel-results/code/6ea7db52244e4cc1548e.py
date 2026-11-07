"""Multi-objective optimizer for WFG4 using deterministic coordinate descent with hypervolume guidance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coordinate descent with hypervolume-guided search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref = np.array([3.0, 5.0, 7.0])

    def _hypervolume_3d(self, points):
        """Compute hypervolume for 3-objective points with reference point [3,5,7]."""
        if len(points) == 0:
            return 0.0
        # Filter points strictly better than reference
        mask = np.all(points < self.ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # For 3D, sort by first objective and compute using sweep
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Remove dominated points from this set for HV computation
        # Keep only non-dominated
        nd_indices = self._non_dominated_indices(pts)
        pts = pts[nd_indices]
        if len(pts) == 0:
            return 0.0
        
        # Sort by obj0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        hv = 0.0
        # For each point, compute the volume it uniquely dominates
        # In 3D: for point i, it dominates the box from pts[i] to ref,
        # but we need to subtract overlapping regions
        # Simple approach: iterate and compute incremental contribution
        for i in range(len(pts)):
            f = pts[i]
            # Box from f to ref
            w0 = self.ref[0] - f[0]
            w1 = self.ref[1] - f[1]
            w2 = self.ref[2] - f[2]
            if w0 <= 0 or w1 <= 0 or w2 <= 0:
                continue
            # Subtract parts already covered by points with smaller obj0
            # Points j < i have smaller obj0, so they cover regions where obj0 < pts[j][0]
            # The region covered by point i that's NOT covered by earlier points:
            # We need to compute the 2D hypervolume in obj1-obj2 space for the slice
            # where obj0 is in [f[0], next_obj0_or_ref)
            
            # Find the next point with larger obj0
            next_obj0 = self.ref[0]
            for j in range(i + 1, len(pts)):
                if pts[j][0] > f[0]:
                    next_obj0 = pts[j][0]
                    break
            
            slice_width = next_obj0 - f[0]
            
            # In the slice [f[0], next_obj0), the dominated region in obj1-obj2 is:
            # Rectangle from (f[1], f[2]) to (ref[1], ref[2])
            # minus any parts dominated by points with obj0 <= f[0] (i.e., points j < i)
            # But those points have obj0 < f[0], so they don't affect this slice directly
            # Actually, points with smaller obj0 may still dominate in obj1, obj2
            # We need: for the 2D projection, what area is NOT dominated by earlier points
            
            # Compute 2D HV in obj1-obj2 for point (f[1], f[2]) with ref (ref[1], ref[2])
            # considering only points that have obj0 < f[0] (they can dominate in this slice)
            earlier = pts[:i]
            if len(earlier) == 0:
                area = (self.ref[1] - f[1]) * (self.ref[2] - f[2])
            else:
                # 2D hypervolume: sort by obj1, sweep
                area = self._hv_2d(f[1], f[2], self.ref[1], self.ref[2], earlier[:, 1], earlier[:, 2])
            
            hv += slice_width * area
        
        return hv

    def _hv_2d(self, f1, f2, ref1, ref2, other_f1, other_f2):
        """Compute 2D hypervolume contribution of point (f1,f2) given other points."""
        # The point dominates rectangle [f1,ref1] x [f2,ref2]
        # Subtract parts dominated by other points
        # A point (g1,g2) dominates part of our rectangle if g1 < f1 and g2 < f2
        # or g1 < f1 and g2 > f2 (partially)
        
        # Actually, the unique contribution of (f1,f2) is the area of [f1,ref1]x[f2,ref2]
        # not covered by any other point's dominated region
        # Another point (g1,g2) covers [g1,ref1]x[g2,ref2]
        # Overlap with our rectangle: [max(f1,g1), ref1] x [max(f2,g2), ref2]
        # if g1 < f1 and g2 < f2: overlap is [f1,ref1]x[f2,ref2] (full)
        # if g1 < f1 and g2 >= f2: overlap is [f1,ref1]x[g2,ref2]
        # if g1 >= f1 and g2 < f2: overlap is [g1,ref1]x[f2,ref2]
        # if g1 >= f1 and g2 >= f2: no overlap (our point dominates them)
        
        # For simplicity, compute total area minus union of overlaps
        # This is complex. Use a simpler approach:
        # The unique area is the area not dominated by any other point.
        # Point (g1,g2) dominates our point if g1<=f1 and g2<=f2 (and at least one strict)
        # If any other point dominates (f1,f2), contribution is 0
        # Otherwise, the contribution is the area of the rectangle minus parts covered by others
        
        # Check if (f1,f2) is dominated by any other
        for j in range(len(other_f1)):
            if other_f1[j] <= f1 and other_f2[j] <= f2 and (other_f1[j] < f1 or other_f2[j] < f2):
                return 0.0
        
        # (f1,f2) is non-dominated among these
        # The unique contribution is the area of [f1,ref1]x[f2,ref2] not covered by other points
        # Other points that don't dominate (f1,f2) but may cover part:
        # Points with g1 < f1 and g2 > f2: cover [f1,ref1]x[g2,ref2]
        # Points with g1 > f1 and g2 < f2: cover [g1,ref1]x[f2,ref2]
        # Points with g1 > f1 and g2 > f2: no cover
        
        # This is getting complex. For a simpler approximation:
        total_area = (ref1 - f1) * (ref2 - f2)
        # Subtract covered areas (approximate by max overlap)
        covered = 0.0
        for j in range(len(other_f1)):
            if other_f1[j] < f1 and other_f2[j] > f2:
                # Covers [f1,ref1] x [other_f2[j], ref2]
                cov = (ref1 - f1) * (ref2 - other_f2[j])
                covered = max(covered, cov)
            elif other_f1[j] > f1 and other_f2[j] < f2:
                # Covers [other_f1[j], ref1] x [f2, ref2]
                cov = (ref1 - other_f1[j]) * (ref2 - f2)
                covered = max(covered, cov)
        
        return max(0.0, total_area - covered)

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return []
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd

    def _hv_contribution(self, f, archive_F):
        """Compute hypervolume contribution of point f to archive."""
        if np.any(f >= self.ref):
            return 0.0
        
        # Check if f is dominated by any point in archive
        for g in archive_F:
            if np.all(g <= f) and np.any(g < f):
                return 0.0
        
        # Compute new HV with f added vs without
        if len(archive_F) == 0:
            if np.all(f < self.ref):
                return (self.ref[0] - f[0]) * (self.ref[1] - f[1]) * (self.ref[2] - f[2])
            return 0.0
        
        # Approximate: compute the box volume minus overlap with archive
        # Simple approximation: if f is non-dominated, estimate its contribution
        # as the volume of the box it uniquely dominates
        # For speed, use a simplified version
        vol = 1.0
        for k in range(3):
            vol *= max(0.0, self.ref[k] - f[k])
        
        # Subtract approximate overlap
        overlap = 0.0
        for g in archive_F:
            if np.any(g >= self.ref):
                continue
            # Overlap box: [max(f[k],g[k]), ref[k]]
            ov = 1.0
            for k in range(3):
                ov *= max(0.0, self.ref[k] - max(f[k], g[k]))
            overlap += ov
        
        # This overcounts overlaps, so use a rough estimate
        return max(0.0, vol * 0.5)  # Conservative estimate

    def _coordinate_descent(self, x0, lo, hi, problem, budget_remaining, rng):
        """Perform coordinate descent with ternary search on each dimension."""
        x = x0.copy()
        f = np.array(problem(x), dtype=np.float64)
        used = 1
        
        # Maintain a small archive for hypervolume guidance
        archive_X = [x.copy()]
        archive_F = [f.copy()]
        
        # Iteratively optimize each coordinate
        max_iters = min(budget_remaining - 1, 60)
        for iteration in range(max_iters):
            if used >= budget_remaining:
                break
            
            improved = False
            for d in range(self.dim):
                if used >= budget_remaining:
                    break
                
                # Ternary search on dimension d
                best_x = x.copy()
                best_f = f.copy()
                best_hv = self._hv_contribution(best_f, archive_F)
                
                # Try several points along dimension d
                # Use a set of candidate values
                candidates = []
                # Current value
                candidates.append(x[d])
                # Some spread values
                mid = (lo[d] + hi[d]) / 2
                candidates.append(mid)
                # Biased toward WFG4 Pareto region for dim 0
                if d == 0:
                    candidates.append(0.35 * (hi[0] - lo[0]) + lo[0])
                    candidates.append(0.30 * (hi[0] - lo[0]) + lo[0])
                    candidates.append(0.40 * (hi[0] - lo[0]) + lo[0])
                # Random samples
                for _ in range(3):
                    candidates.append(lo[d] + (hi[d] - lo[d]) * rng.random())
                
                # Evaluate candidates
                for c in candidates:
                    if used >= budget_remaining:
                        break
                    c = np.clip(c, lo[d], hi[d])
                    if abs(c - x[d]) < 1e-10:
                        continue
                    test_x = x.copy()
                    test_x[d] = c
                    test_f = np.array(problem(test_x), dtype=np.float64)
                    used += 1
                    
                    # Check if this improves hypervolume
                    test_hv = self._hv_contribution(test_f, archive_F)
                    if test_hv > best_hv:
                        best_hv = test_hv
                        best_x = test_x.copy()
                        best_f = test_f.copy()
                        improved = True
                
                if improved:
                    x = best_x
                    f = best_f
                    # Update archive
                    if not any(np.all(g <= f) and np.any(g < f) for g in archive_F):
                        archive_F.append(f.copy())
                        archive_X.append(x.copy())
                        # Trim archive if too large
                        if len(archive_F) > 20:
                            nd_idx = self._non_dominated_indices(np.array(archive_F))
                            archive_F = [archive_F[i] for i in nd_idx]
                            archive_X = [archive_X[i] for i in nd_idx]
        
        return x, f, archive_X, archive_F, used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Global archive
        global_archive_X = []
        global_archive_F = []

        # Multiple restarts with diverse initializations
        n_restarts = 8
        per_restart_budget = budget // n_restarts

        for r in range(n_restarts):
            if used >= budget or time.time() > deadline:
                break
            
            # Initialize with diversity, biasing dim 0 toward WFG4 Pareto region
            x0 = lo + (hi - lo) * rng.random_sample(dim)
            if r == 0:
                # First restart: bias toward known Pareto region
                x0[0] = lo[0] + 0.35 * (hi[0] - lo[0])
            elif r < 4:
                # Some bias for early restarts
                x0[0] = lo[0] + (0.25 + 0.1 * rng.random()) * (hi[0] - lo[0])
            
            # Run coordinate descent
            x_best, f_best, arc_X, arc_F, used_here = self._coordinate_descent(
                x0, lo, hi, problem, per_restart_budget - used, rng
            )
            used += used_here
            
            # Add to global archive
            for ax, af in zip(arc_X, arc_F):
                if not any(np.all(g <= af) and np.any(g < af) for g in global_archive_F):
                    global_archive_F.append(af)
                    global_archive_X.append(ax)
            
            # Trim global archive
            if len(global_archive_F) > 50:
                arr_F = np.array(global_archive_F)
                nd_idx = self._non_dominated_indices(arr_F)
                global_archive_F = [global_archive_F[i] for i in nd_idx]
                global_archive_X = [global_archive_X[i] for i in nd_idx]
        
        # Use remaining budget for local refinement of best points
        remaining = budget - used
        if remaining > 5 and len(global_archive_X) > 0 and time.time() < deadline:
            # Pick the point with best hypervolume contribution and refine
            arr_F = np.array(global_archive_F)
            best_idx = 0
            best_hv = -1
            for i in range(len(arr_F)):
                hv = self._hv_contribution(arr_F[i], [arr_F[j] for j in range(len(arr_F)) if j != i])
                if hv > best_hv:
                    best_hv = hv
                    best_idx = i
            
            x_refine = global_archive_X[best_idx].copy()
            f_refine = np.array(problem(x_refine), dtype=np.float64)
            used += 1
            
            # Quick coordinate refinement
            for d in range(dim):
                if used >= budget or time.time() > deadline:
                    break
                for _ in range(2):
                    if used >= budget or time.time() > deadline:
                        break
                    step = (hi[d] - lo[d]) * 0.1 * rng.random()
                    for sign in [1, -1]:
                        if used >= budget or time.time() > deadline:
                            break
                        test_x = x_refine.copy()
                        test_x[d] = np.clip(test_x[d] + sign * step, lo[d], hi[d])
                        test_f = np.array(problem(test_x), dtype=np.float64)
                        used += 1
                        if self._hv_contribution(test_f, global_archive_F) > 0:
                            x_refine = test_x
                            f_refine = test_f
        
        # Return the archive points
        if len(global_archive_X) > 0:
            return np.array(global_archive_X)
        return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
