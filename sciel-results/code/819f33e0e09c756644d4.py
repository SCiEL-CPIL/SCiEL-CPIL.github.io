"""Multi-objective optimizer for RE21 with hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead hypervolume-guided search for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([3000.0, 0.06])

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if time.time() > deadline:
                return None
            fv = problem(np.asarray(x, dtype=float))
            used += 1
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(fv, dtype=float))
            return fv

        def compute_hv_contribution(new_point, existing_points):
            """Compute hypervolume contribution of new_point given existing non-dominated set."""
            # Filter existing points that are better than reference
            mask = np.all(existing_points < ref, axis=1)
            pts = existing_points[mask]
            
            # Add new point if it's also better than reference
            if np.all(new_point < ref):
                pts = np.vstack([pts, new_point.reshape(1, -1)])
            
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # Compute HV without the new point (if it was added)
            def hv_2d(points):
                if len(points) == 0:
                    return 0.0
                points = points[np.argsort(points[:, 0])]
                hv = 0.0
                max_y = ref[1]
                for i in range(len(points)):
                    x = points[i, 0]
                    y = points[i, 1]
                    if y < max_y:
                        hv += (ref[0] - x) * (max_y - y)
                        max_y = y
                return hv
            
            # HV with new point
            hv_with = hv_2d(pts)
            
            # HV without new point
            if np.all(new_point < ref):
                pts_without = pts[:-1] if len(pts) > 1 else np.empty((0, 2))
                hv_without = hv_2d(pts_without)
            else:
                hv_without = hv_2d(pts)
            
            return hv_with - hv_without

        def get_non_dominated_indices(F):
            """Get indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Strategy: Multi-start Nelder-Mead from stratified random points
        # Phase 1: Stratified initialization to cover the search space
        n_starts = min(20, budget // 10)  # 20 starting points, each gets ~10 evals
        rng = np.random.default_rng(42)
        
        # Create stratified grid in objective space conceptually
        # For 4D, use Latin hypercube sampling
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=dim, seed=42)
        stratified = sampler.random(n=n_starts)
        X_starts = lo + stratified * (hi - lo)
        
        # Evaluate starting points
        F_starts = []
        for i in range(n_starts):
            if time.time() > deadline or used >= budget:
                break
            fv = evaluate(X_starts[i])
            if fv is not None:
                F_starts.append(fv)
        
        if len(F_starts) == 0:
            return np.array([])
        
        F_starts = np.array(F_starts)
        
        # Phase 2: Nelder-Mead local search from each start
        # For each start, do a few Nelder-Mead iterations
        nm_budget_per_start = max(3, (budget - used) // max(1, n_starts))
        
        for s in range(n_starts):
            if time.time() > deadline or used >= budget:
                break
            
            # Initialize simplex around the starting point
            x0 = X_starts[s].copy()
            f0 = F_starts[s]
            
            # Build initial simplex
            simplex_X = [x0.copy()]
            simplex_F = [f0.copy()]
            
            # Add perturbed points
            for d in range(dim):
                if time.time() > deadline or used >= budget:
                    break
                x_pert = x0.copy()
                step = 0.05 * (hi[d] - lo[d])
                x_pert[d] += step
                x_pert = np.clip(x_pert, lo, hi)
                fv = evaluate(x_pert)
                if fv is not None:
                    simplex_X.append(x_pert.copy())
                    simplex_F.append(fv.copy())
            
            if len(simplex_X) < 2:
                continue
            
            # Nelder-Mead iterations
            max_nm_iters = max(2, nm_budget_per_start - len(simplex_X))
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for _ in range(max_nm_iters):
                if time.time() > deadline or used >= budget:
                    break
                
                n_simplex = len(simplex_X)
                if n_simplex < 2:
                    break
                
                # Sort by hypervolume contribution (we want to minimize loss)
                # Use sum of objectives as proxy for now, or actual HV contribution
                # For efficiency, use a simple scalarization
                sorted_idx = np.argsort(np.sum(simplex_F, axis=1))
                simplex_X = [simplex_X[i] for i in sorted_idx]
                simplex_F = [simplex_F[i] for i in sorted_idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex_X[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex_X[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = evaluate(x_r)
                if f_r is None:
                    break
                
                # Check if reflection improves HV
                # Replace worst if reflection is better
                if np.sum(f_r) < np.sum(simplex_F[-1]):
                    simplex_X[-1] = x_r
                    simplex_F[-1] = f_r
                    
                    # Try expansion
                    if np.sum(f_r) < np.sum(simplex_F[0]):
                        x_e = centroid + gamma * (centroid - simplex_X[-1])
                        x_e = np.clip(x_e, lo, hi)
                        f_e = evaluate(x_e)
                        if f_e is not None and np.sum(f_e) < np.sum(f_r):
                            simplex_X[-1] = x_e
                            simplex_F[-1] = f_e
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex_X[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = evaluate(x_c)
                    if f_c is not None:
                        if np.sum(f_c) < np.sum(simplex_F[-1]):
                            simplex_X[-1] = x_c
                            simplex_F[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex_X)):
                                if time.time() > deadline or used >= budget:
                                    break
                                simplex_X[i] = simplex_X[0] + sigma * (simplex_X[i] - simplex_X[0])
                                simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                                f_s = evaluate(simplex_X[i])
                                if f_s is not None:
                                    simplex_F[i] = f_s

        # Phase 3: Final refinement - find best hypervolume-contributing points
        if used < budget and time.time() < deadline:
            # Get current non-dominated set
            if all_F:
                F_arr = np.array(all_F)
                nd_idx = get_non_dominated_indices(F_arr)
                if nd_idx:
                    nd_F = F_arr[nd_idx]
                    nd_X = [all_X[i] for i in nd_idx]
                    
                    # Try to improve by exploring near the ideal point
                    ideal = np.min(nd_F, axis=0)
                    # Interpolate between best points
                    for i in range(len(nd_X)):
                        if time.time() > deadline or used >= budget:
                            break
                        for j in range(i + 1, len(nd_X)):
                            if time.time() > deadline or used >= budget:
                                break
                            for t in [0.25, 0.5, 0.75]:
                                if time.time() > deadline or used >= budget:
                                    break
                                x_interp = t * nd_X[i] + (1 - t) * nd_X[j]
                                x_interp = np.clip(x_interp, lo, hi)
                                fv = evaluate(x_interp)
                                if fv is None:
                                    break

        return np.array(all_X) if all_X else np.array([])
# EVOLVE-BLOCK-END
