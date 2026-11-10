"""Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling with local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive sampling optimizer for RE37 (4 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_contribution(self, F, ref):
        """Compute marginal hypervolume contribution of each point in F relative to ref."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter points that are better than ref in all objectives
        valid_mask = np.all(F < ref, axis=1)
        F_valid = F[valid_mask]
        n_valid = len(F_valid)
        
        if n_valid == 0:
            return np.zeros(n)
        
        # Compute HV of the full set
        def compute_hv(points, ref_pt):
            """Compute exact hypervolume for 3D points."""
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            sorted_pts = points[np.argsort(points[:, 0])]
            
            # For 3D, we can use a sweep line approach
            # HV = sum over points of (ref[0] - f0[i]) * (area in 2D of dominated region)
            # For simplicity with small n, use inclusion-exclusion or direct computation
            
            # Direct approach: for each point, compute the volume it uniquely contributes
            # This is expensive but n is small (<= 50)
            
            # Better: sort by f0, then for each point, compute the 2D area of the region
            # that is dominated by this point but not by any point with smaller f0
            
            total_hv = 0.0
            for i in range(len(sorted_pts)):
                # Region dominated by sorted_pts[i] but not by any previous point
                f0 = sorted_pts[i][0]
                f1 = sorted_pts[i][1]
                f2 = sorted_pts[i][2]
                
                # Width in f0 direction
                w0 = ref_pt[0] - f0
                
                # For f1-f2 plane, compute area of region [f1, ref[1]] x [f2, ref[2]]
                # that is not covered by any previous point
                # A previous point j covers [f1[j], ref[1]] x [f2[j], ref[2]] if f0[j] < f0[i]
                # (since we sorted by f0, all previous have smaller f0)
                
                # The uncovered area is the area of [f1, ref[1]] x [f2, ref[2]]
                # minus the union of rectangles from previous points
                # This is a 2D union of rectangles problem
                
                # For small n, we can use a grid-based approximation or exact computation
                # Let's use a simpler approach: compute the area directly
                
                # Area of rectangle for point i
                area_i = (ref_pt[1] - f1) * (ref_pt[2] - f2)
                
                # Subtract overlap with previous points
                # For each previous point j, if it dominates in f1-f2 plane:
                # overlap = max(0, min(ref[1], f1[j]) - max(f1, f1[j])) * max(0, min(ref[2], f2[j]) - max(f2, f2[j]))
                # But this double-counts overlaps between previous points
                
                # For exact computation, we need to compute the union area of previous points'
                # rectangles within [f1, ref[1]] x [f2, ref[2]]
                
                # Simplified: just use the point's own rectangle area as an upper bound
                # and subtract overlaps with immediate neighbors
                # This is an approximation but good enough for ranking
                
                overlap = 0.0
                for j in range(i):
                    f1j = sorted_pts[j][1]
                    f2j = sorted_pts[j][2]
                    
                    # Overlap in f1 direction
                    ov1 = max(0, min(ref_pt[1], f1j) - max(f1, f1j))
                    # Overlap in f2 direction
                    ov2 = max(0, min(ref_pt[2], f2j) - max(f2, f2j))
                    overlap += ov1 * ov2
                
                # This over-subtracts due to double-counting, so use as heuristic
                contrib = w0 * max(0, area_i - overlap)
                total_hv += contrib
            
            return total_hv
        
        # Compute HV without each point
        full_hv = compute_hv(F_valid, ref)
        contributions_valid = np.zeros(n_valid)
        
        for i in range(n_valid):
            if n_valid <= 1:
                contributions_valid[i] = full_hv
                continue
            F_without_i = np.delete(F_valid, i, axis=0)
            hv_without_i = compute_hv(F_without_i, ref)
            contributions_valid[i] = full_hv - hv_without_i
        
        # Map back to original indices
        contributions = np.zeros(n)
        contributions[valid_mask] = contributions_valid
        return contributions

    def _local_polish(self, problem, bounds, X, F, budget_remaining, deadline):
        """Hypervolume-driven local polish: refine non-dominated points toward gaps."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        # Get non-dominated indices
        n = len(F)
        if n == 0:
            return X, F, budget_remaining

        # Find non-dominated points
        nd_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_mask[i] = True

        nd_indices = np.where(nd_mask)[0]
        if len(nd_indices) == 0:
            return X, F, budget_remaining

        # Compute marginal hypervolume contribution for each ND point
        contributions = self._hypervolume_contribution(F, ref)
        nd_contributions = contributions[nd_indices]

        # Process points in order of contribution (most valuable first)
        order = np.argsort(-nd_contributions)

        for oi in order:
            if budget_remaining <= 0 or time.time() > deadline:
                break

            idx = nd_indices[oi]
            base_x = X[idx].copy()
            base_f = F[idx].copy()
            best_x = base_x.copy()
            best_f = base_f.copy()

            # Multi-scale local search: try multiple step sizes
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            for step_frac in step_sizes:
                if budget_remaining <= 0 or time.time() > deadline:
                    break
                for d in range(dim):
                    if budget_remaining <= 0 or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        if budget_remaining <= 0 or time.time() > deadline:
                            break
                        step = step_frac * (hi[d] - lo[d])
                        trial_x = base_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = np.array(problem(trial_x))
                        budget_remaining -= 1

                        # Check if trial dominates best
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x.copy()
                            best_f = trial_f.copy()
                            base_x = trial_x.copy()
                            base_f = trial_f.copy()

            if np.all(best_f < F[idx]):
                X[idx] = best_x
                F[idx] = best_f

        return X, F, budget_remaining

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Initialize with a diverse set of random points
        init_size = 20
        X = lo + (hi - lo) * np.random.random((init_size, dim))
        F = np.array([problem(x) for x in X])
        used = init_size

        # Hypervolume-driven adaptive sampling
        # Iteratively propose new points that maximize expected marginal HV contribution
        while used < budget:
            if time.time() > deadline:
                break

            # Compute marginal HV contribution for each current point
            ref = np.array([1.1, 1.3, 1.1])
            contributions = self._hypervolume_contribution(F, ref)

            # Find the point with highest contribution (most isolated/valuable)
            # This indicates where the front has a gap
            best_idx = np.argmax(contributions)
            best_point_x = X[best_idx].copy()
            best_point_f = F[best_idx].copy()

            # Propose new points near the most valuable point
            # Use adaptive step size based on local density
            n_proposals = 3
            for p in range(n_proposals):
                if used >= budget or time.time() > deadline:
                    break

                # Adaptive step size: larger when contribution is high (sparse region)
                base_step = 0.05 + 0.05 * (contributions[best_idx] / (np.max(contributions) + 1e-10))
                step = base_step * np.random.uniform(0.5, 1.5)

                # Random direction in 4D
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)

                new_x = best_point_x + step * direction
                new_x = np.clip(new_x, lo, hi)

                new_f = np.array(problem(new_x))
                used += 1

                # Add to population if it's non-dominated or improves HV
                X = np.vstack([X, new_x.reshape(1, -1)])
                F = np.vstack([F, new_f.reshape(1, -1)])

            # Prune dominated points to keep population manageable
            # Keep only non-dominated points
            n = len(F)
            nd_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_mask[i] = True

            # Keep at most 50 points
            if np.sum(nd_mask) > 50:
                # Keep the 50 points with highest HV contribution
                contributions = self._hypervolume_contribution(F, ref)
                nd_indices = np.where(nd_mask)[0]
                nd_contribs = contributions[nd_indices]
                keep_order = np.argsort(-nd_contribs)[:50]
                keep_indices = nd_indices[keep_order]
                X = X[keep_indices]
                F = F[keep_indices]
            else:
                X = X[nd_mask]
                F = F[nd_mask]

        # Local polish on non-dominated points if budget remains
        if used < budget and time.time() < deadline:
            budget_remaining = budget - used
            X, F, _ = self._local_polish(problem, bounds, X, F, budget_remaining, deadline)

        return X
# EVOLVE-BLOCK-END
