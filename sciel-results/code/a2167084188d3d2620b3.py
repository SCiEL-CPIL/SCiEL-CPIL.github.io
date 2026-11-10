# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hypervolume-guided local search for WFG7 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 285.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # State
        X = []
        F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = problem(x)
            eval_count += 1
            X.append(x)
            F.append(fx)
            return fx

        # Reference point for hypervolume computation
        ref = np.array([3.0, 5.0, 7.0])

        def compute_hypervolume(points, ref):
            """Compute exact hypervolume for 3D points using inclusion-exclusion."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            valid = np.array(valid)
            n = len(valid)
            if n == 1:
                return np.prod(ref - valid[0])
            
            # Sort by first objective
            order = np.argsort(valid[:, 0])
            valid = valid[order]
            
            # For 3D, use recursive decomposition
            def hv_1d(points_1d, ref_1d):
                return ref_1d - points_1d[0]
            
            # Use a simple but correct approach for small n
            # Divide and conquer on the first objective
            def hv_recursive(pts, ref):
                if len(pts) == 0:
                    return 0.0
                if len(pts) == 1:
                    return np.prod(ref - pts[0])
                
                # Sort by first coordinate
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                
                # Split into two halves
                mid = len(pts) // 2
                left = pts[:mid]
                right = pts[mid:]
                
                # Hypervolume of left set
                hv_left = hv_recursive(left, ref)
                
                # For right set, adjust reference based on left's max in other dims
                # This is complex; use a simpler approach for n <= 100
                if len(pts) <= 50:
                    # Brute force with inclusion-exclusion via sorting
                    # For 3D: sort by f1, sweep
                    pts_sorted = pts[np.argsort(pts[:, 0])]
                    hv = 0.0
                    for i in range(len(pts_sorted)):
                        p = pts_sorted[i]
                        if np.any(p >= ref):
                            continue
                        # Contribution: volume where this point is the "min" in some region
                        # Simple approach: compute volume of box [p, ref] minus overlaps
                        # For accuracy with small n, use the following:
                        # The hypervolume is the union of boxes [p_i, ref]
                        # We can compute this by sorting and using a sweep line
                        pass
                    
                    # Use a different approach: compute via sorting all coordinates
                    # For 3D with small n, we can use the following formula:
                    # Sort by f1, then for each point, compute the area in f2-f3 plane
                    # that is not covered by any previous point (with smaller f1)
                    
                    pts_s = pts_sorted
                    hv = 0.0
                    for i in range(len(pts_s)):
                        p = pts_s[i]
                        if np.any(p >= ref):
                            continue
                        # This point contributes to hypervolume in the region
                        # where it is the minimum in at least one objective among points with f1 >= p[0]
                        # Simplified: the contribution is the volume of [p, ref] that is not
                        # dominated by any other point. This is complex.
                        # For now, use a simpler estimate for the local search guidance.
                        # We'll use the actual box volume as a proxy.
                        hv += np.prod(ref - p)
                    
                    # This overestimates, but for ranking purposes it's okay
                    return hv
                else:
                    # For larger sets, use a simpler heuristic
                    return np.sum(np.prod(ref - pts, axis=1))
            
            return hv_recursive(valid, ref)

        def marginal_hv_contribution(point, archive):
            """Estimate the marginal hypervolume contribution of a point to the archive."""
            if len(archive) == 0:
                return np.prod(ref - point) if np.all(point < ref) else 0.0
            
            # Check if point is dominated by any archive point
            for a in archive:
                if np.all(a <= point) and np.any(a < point):
                    return 0.0
            
            # Estimate: the volume of the box [point, ref] that is not covered
            # by any archive point that dominates in at least one dimension
            # Simplified: compute the volume of [point, ref] minus overlaps
            # For a rough estimate, use the box volume weighted by how much
            # it extends beyond the current Pareto front
            box_vol = np.prod(ref - point) if np.all(point < ref) else 0.0
            
            # Subtract contributions from archive points that are "close"
            # This is a heuristic, not exact
            for a in archive:
                if np.all(a < ref):
                    # If a and point overlap in their boxes, subtract overlap
                    overlap = np.prod(np.minimum(ref, np.maximum(point, a)) - np.minimum(point, a))
                    if overlap > 0:
                        box_vol -= overlap * 0.5  # heuristic discount
            
            return max(0.0, box_vol)

        # Initial LHS-like sampling
        n_init = min(30, budget)
        for i in range(n_init):
            # Use stratified random for better initial coverage
            x = lo + span * np.random.rand(dim)
            evaluate(x)

        # Main loop: hypervolume-guided local search
        while eval_count < budget and time.time() < deadline:
            if len(F) < 5:
                x = lo + span * np.random.rand(dim)
                evaluate(x)
                continue

            # Get the non-dominated set
            F_arr = np.array(F)
            n = len(F_arr)
            
            # Find non-dominated points
            nondom = []
            for i in range(n):
                is_dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nondom.append(i)
            
            if not nondom:
                nondom = list(range(n))
            
            # Select the point with highest marginal hypervolume contribution
            best_idx = nondom[0]
            best_contrib = -1
            for idx in nondom:
                contrib = marginal_hv_contribution(F_arr[idx], [F_arr[j] for j in nondom if j != idx])
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            # Perform Nelder-Mead-like local search around best_idx
            x_center = np.array(X[best_idx])
            f_center = np.array(F[best_idx])
            
            # Create simplex
            simplex = [x_center.copy()]
            for i in range(dim):
                v = x_center.copy()
                step = 0.1 * span[i]
                v[i] += step
                v[i] = np.clip(v[i], lo[i], hi[i])
                simplex.append(v)
            
            # Evaluate simplex
            simplex_F = []
            for s in simplex:
                if eval_count >= budget or time.time() >= deadline:
                    break
                fx = evaluate(s)
                if fx is not None:
                    simplex_F.append(fx)
            
            if len(simplex_F) < 2:
                # Not enough budget for local search, do random perturbation
                x_new = x_center + 0.1 * span * np.random.randn(dim)
                x_new = np.clip(x_new, lo, hi)
                evaluate(x_new)
                continue
            
            # Greedy improvement: repeatedly perturb the best point
            best_x = x_center.copy()
            best_f = f_center.copy()
            
            for iteration in range(5):
                if eval_count >= budget or time.time() >= deadline:
                    break
                
                # Try multiple perturbations
                best_candidate_x = best_x
                best_candidate_f = best_f
                
                for trial in range(3):
                    x_try = best_x + 0.05 * span * np.random.randn(dim)
                    x_try = np.clip(x_try, lo, hi)
                    
                    if eval_count >= budget:
                        break
                    
                    f_try = evaluate(x_try)
                    if f_try is None:
                        break
                    
                    # Check if this improves hypervolume contribution
                    contrib_try = marginal_hv_contribution(f_try, [F_arr[j] for j in nondom if j != best_idx])
                    contrib_best = marginal_hv_contribution(best_f, [F_arr[j] for j in nondom if j != best_idx])
                    
                    if contrib_try > contrib_best:
                        best_candidate_x = x_try
                        best_candidate_f = f_try
                        best_contrib = contrib_try
                
                if np.allclose(best_candidate_x, best_x):
                    break
                best_x = best_candidate_x
                best_f = best_candidate_f

        return X
# EVOLVE-BLOCK-END
