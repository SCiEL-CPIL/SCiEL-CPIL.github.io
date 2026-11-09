"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided parent selection with adaptive step size."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget
        dim = self.dim

        # Initialize with random points
        X = lo + (hi - lo) * rng.random((n, dim))
        F = np.array([problem(x) for x in X])
        used = n

        # Reference point for HV computation
        ref = np.array([1705.0, 12.0, 0.3])

        def compute_hv_contribution(F_mat, ref_pt):
            """Compute hypervolume contribution of a set of points in 3D."""
            # Filter points strictly better than reference
            mask = np.all(F_mat < ref_pt, axis=1)
            F_mat = F_mat[mask]
            if len(F_mat) == 0:
                return 0.0

            # For 3D HV, sort by first objective
            idx = np.argsort(F_mat[:, 0])
            F_mat = F_mat[idx]

            hv = 0.0
            # Process points in order of first objective
            # For each point, compute the volume of the region it uniquely covers
            # In 3D: for point i, the contribution is:
            #   (F[i,0] - F[i+1,0] if i < n-1 else ref[0] - F[i,0]) * 
            #   max(0, ref[1] - F[i,1]) * max(0, ref[2] - F[i,2])
            # But this is only correct if points are sorted and we account for dominance properly.
            # A more accurate approach: sort by obj0, then for each point compute the 2D HV in obj1-obj2
            # of the region where obj0 < current point's obj0.

            # Simpler approach for small sets: use inclusion-exclusion or just compute
            # the union of boxes. For n=200, this is O(n^2) which is fine.
            
            # Actually, let's use a proper 3D HV algorithm.
            # Sort by first coordinate
            n_pts = len(F_mat)
            if n_pts == 1:
                d0 = max(0.0, ref_pt[0] - F_mat[0, 0])
                d1 = max(0.0, ref_pt[1] - F_mat[0, 1])
                d2 = max(0.0, ref_pt[2] - F_mat[0, 2])
                return d0 * d1 * d2

            # Sort by first objective
            order = np.argsort(F_mat[:, 0])
            F_sorted = F_mat[order]

            # For each point in sorted order, compute the 2D hypervolume in (obj1, obj2)
            # of the region dominated by this point but not by any subsequent point
            hv = 0.0
            for i in range(n_pts):
                # Width in obj0 direction
                if i + 1 < n_pts:
                    w0 = F_sorted[i + 1, 0] - F_sorted[i, 0]
                else:
                    w0 = ref_pt[0] - F_sorted[i, 0]
                w0 = max(0.0, w0)
                if w0 <= 0:
                    continue

                # 2D HV in (obj1, obj2) for points from i to end
                # A point j >= i contributes to the 2D region if it doesn't dominate in obj1 and obj2
                # The 2D HV of the set {F_sorted[i:], ref[1:], } in 2D
                # But we need the region that is dominated by point i but not by any other point in the set
                # Actually, for the contribution of point i in the sorted-by-obj0 list:
                # The region is: obj0 in [F[i,0], next_obj0), and (obj1, obj2) in the 2D region
                # dominated by (F[i,1], F[i,2]) but NOT dominated by any (F[j,1], F[j,2]) for j > i
                # Wait, that's not quite right either. Let me think again.
                
                # The standard approach: sort by obj0. For point i, the volume it uniquely contributes
                # is the 2D hypervolume in (obj1, obj2) of the set of points {i, i+1, ..., n-1} 
                # with reference (ref[1], ref[2]), times the width in obj0.
                # But that's not right either because later points may have smaller obj1/obj2.
                
                # Correct approach: The contribution of point i is:
                # w0 * HV_2D({(F[j,1], F[j,2]) for j >= i}, (ref[1], ref[2]))
                # No, that's also not right.
                
                # Let me use a different approach. The total HV is the union of boxes.
                # For small n, I'll just compute it properly.
                
                pass

            # Let me just use a simple but correct O(n^2) 3D HV computation
            # For each point, compute the box it defines, then subtract overlaps
            # This is complex. Let me use a different strategy.
            
            # Alternative: since we only need to compare HV values for parent selection,
            # and we're working with ~200 points, let me use a simpler heuristic.
            # The HV can be approximated by summing individual contributions with 
            # proper dominance filtering.
            
            # Actually, for parent selection, I don't need exact HV. I can use a 
            # simpler metric: the sum of individual "contributions" where each point's
            # contribution is the volume of its box minus what's covered by dominated points.
            
            # Let me just compute exact HV using a recursive approach for 3D.
            return self._hv_3d(F_mat, ref_pt)

        def _hv_3d(self, F_mat, ref_pt):
            """Compute exact 3D hypervolume."""
            n_pts = len(F_mat)
            if n_pts == 0:
                return 0.0
            if n_pts == 1:
                d0 = max(0.0, ref_pt[0] - F_mat[0, 0])
                d1 = max(0.0, ref_pt[1] - F_mat[0, 1])
                d2 = max(0.0, ref_pt[2] - F_mat[0, 2])
                return d0 * d1 * d2

            # Sort by first objective
            order = np.argsort(F_mat[:, 0])
            F_sorted = F_mat[order]

            hv = 0.0
            for i in range(n_pts):
                # Width in obj0
                if i + 1 < n_pts:
                    w0 = F_sorted[i + 1, 0] - F_sorted[i, 0]
                else:
                    w0 = ref_pt[0] - F_sorted[i, 0]
                w0 = max(0.0, w0)
                if w0 <= 0:
                    continue

                # 2D HV of points i..n-1 in (obj1, obj2) space with reference (ref[1], ref[2])
                # But only count the part dominated by point i's (obj1, obj2)
                # Actually, the correct formula: the contribution of point i is w0 * HV_2D of 
                # the set {(F[j,1], F[j,2]) : j >= i} with reference (ref[1], ref[2])
                # No wait, that would double count.
                
                # The correct approach: sort by obj0. The HV is the sum over i of:
                #   (obj0_{i+1} - obj0_i) * HV_2D({(obj1_j, obj2_j) : j >= i}, (ref1, ref2))
                # where obj0_{n} = ref0
                # This works because for each "slice" in obj0 between point i and i+1,
                # the cross-section is the 2D HV of all points with obj0 <= that slice value.
                
                # So I need to compute 2D HV for suffixes. Let me do it incrementally.
                # For each i, compute 2D HV of F_sorted[i:, 1:3] with ref (ref[1], ref[2])
                
                sub = F_sorted[i:, 1:3]
                ref2 = ref_pt[1:3]
                hv2 = self._hv_2d(sub, ref2)
                hv += w0 * hv2

            return hv

        def _hv_2d(self, F_mat, ref_pt):
            """Compute exact 2D hypervolume."""
            n_pts = len(F_mat)
            if n_pts == 0:
                return 0.0
            if n_pts == 1:
                d0 = max(0.0, ref_pt[0] - F_mat[0, 0])
                d1 = max(0.0, ref_pt[1] - F_mat[0, 1])
                return d0 * d1

            # Sort by first coordinate
            order = np.argsort(F_mat[:, 0])
            F_sorted = F_mat[order]

            hv = 0.0
            for i in range(n_pts):
                if i + 1 < n_pts:
                    w0 = F_sorted[i + 1, 0] - F_sorted[i, 0]
                else:
                    w0 = ref_pt[0] - F_sorted[i, 0]
                w0 = max(0.0, w0)
                if w0 <= 0:
                    continue
                # Height is max(0, ref[1] - min of obj1 for j >= i)
                # Actually, the 2D HV contribution of point i is:
                # w0 * max(0, ref[1] - min(F_sorted[i:, 1]))
                # No, that's not right. It's w0 * (ref[1] - F_sorted[i, 1]) if F_sorted[i,1] < ref[1]
                # But only if no later point has a smaller obj1.
                
                # Correct: sort by obj0. For point i, the contribution is:
                # w0 * max(0, ref[1] - min(F_sorted[i:, 1]))
                # Because the "height" at this slice is determined by the best (smallest) obj1 
                # among all points with obj0 <= current slice.
                
                min_obj1 = np.min(F_sorted[i:, 1])
                h = max(0.0, ref_pt[1] - min_obj1)
                hv += w0 * h

            return hv

        # Adaptive step size
        step_size = 0.15
        min_step = 0.02
        max_step = 0.4
        improvement_count = 0

        while used < n:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Compute HV contributions for parent selection
            # Use a fast approximation: for each point, compute its "niche" contribution
            # We'll use the exact HV but only for non-dominated points to save time
            
            # Find non-dominated points
            n_pts = len(F)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if is_dominated[i]:
                    continue
                # Check if any other point dominates i
                # F[j] <= F[i] and F[j] != F[i]
                dominates = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominates[i] = False  # Don't count self
                if np.any(dominates):
                    is_dominated[i] = True

            nd_idx = np.where(~is_dominated)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(n_pts)

            # For parent selection, use HV contribution
            # Compute HV for each non-dominated point's contribution
            # To save time, compute approximate contribution:
            # For each ND point, compute the box volume and subtract overlap with other ND points
            
            # Simpler: use the 2D/3D HV approach but only for ND points
            # For efficiency with 200 points, let's use a simpler metric
            # that correlates with HV contribution: the "volume" of the region
            # uniquely dominated by each point.
            
            # Compute contributions using the sorted approach
            # But this is O(n^2) per iteration which might be slow.
            # Let's use a simpler heuristic: normalize objectives and use 
            # the distance to the ideal point as a proxy.
            
            # Actually, let me compute exact HV contributions for ND points only.
            # With ~20-50 ND points, this is fast.
            
            F_nd = F[nd_idx]
            
            # Compute HV contribution of each point
            # For each point i, its contribution is the HV of {i} minus HV of others
            # That's O(n^2) HV computations which is too slow.
            
            # Alternative: use the "niche" approach - for each point, compute the 
            # volume of the box it defines, weighted by how many other points dominate parts of it.
            
            # Simplest effective approach: use the product of distances to reference
            # as a proxy for HV contribution, normalized.
            
            # Let me just use a random selection from ND points with some bias.
            # Or better: use the HV-based approach but compute it once per iteration.
            
            # For speed, let me use a simpler parent selection:
            # 1. Find ND points
            # 2. For each ND point, compute a "score" based on its position
            # 3. Select parent with probability proportional to score
            
            # Score: for each ND point, compute the volume of the box it dominates
            # that isn't dominated by other ND points. Approximate by:
            #   prod(ref - F[i]) * (1 / (1 + number of ND points closer to ideal))
            
            # Let me just use the exact approach but limit to ND points
            # and accept the O(k^2) cost where k = number of ND points
            
            if len(nd_idx) <= 50:
                # Compute HV contribution for each ND point
                contributions = np.zeros(len(nd_idx))
                for i in range(len(nd_idx)):
                    # HV of all ND points
                    hv_all = self._hv_3d(F_nd, ref)
                    # HV without point i
                    if len(F_nd) > 1:
                        mask = np.ones(len(F_nd), dtype=bool)
                        mask[i] = False
                        hv_without = self._hv_3d(F_nd[mask], ref)
                    else:
                        hv_without = 0.0
                    contributions[i] = hv_all - hv_without
                
                # Normalize
                total = contributions.sum()
                if total > 0:
                    probs = contributions / total
                else:
                    probs = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                # Too many ND points, use simple heuristic
                # Score based on distance to ideal (lower is better)
                scores = np.prod(ref - F_nd, axis=1)
                scores = np.maximum(scores, 0)
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                else:
                    probs = np.ones(len(nd_idx)) / len(nd_idx)

            # Select parent
            parent_idx = nd_idx[rng.choice(len(nd_idx), p=probs)]
            parent = X[parent_idx]

            # Adaptive step size
            # If we're making progress, keep step size; otherwise reduce
            child = parent + step_size * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point (highest sum of normalized objectives)
            # Use HV-based worst selection: remove the point with lowest HV contribution
            # For speed, use sum of objectives as proxy
            # Actually, let's use a simple replacement: replace the point that is 
            # dominated by the most other points, or the one with highest objective sum
            
            # Simple: replace the point with the highest sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            
            # Check if child improves HV
            F_new = F.copy()
            F_new[worst] = fc
            X_new = X.copy()
            X_new[worst] = child
            
            # Quick check: is the child non-dominated?
            child_dominated = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
            
            if not child_dominated:
                # Child is non-dominated, accept
                X[worst] = child
                F[worst] = fc
                improvement_count += 1
            else:
                # Child is dominated, still accept if it's better than the worst point
                # in terms of objective sum
                if np.sum(fc) < np.sum(F[worst]):
                    X[worst] = child
                    F[worst] = fc
                    improvement_count += 1
                else:
                    # Reject, reduce step size
                    step_size = max(min_step, step_size * 0.9)

            # Adapt step size
            if improvement_count % 10 == 0 and improvement_count > 0:
                # Occasionally increase step size for exploration
                if rng.random() < 0.3:
                    step_size = min(max_step, step_size * 1.2)

            # Reset improvement count periodically
            if used % 50 == 0:
                improvement_count = 0

        return X
# EVOLVE-BLOCK-END
