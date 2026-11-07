"""Surrogate-assisted multi-objective optimizer using GP + EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """GP-based multi-objective optimizer using Expected Hypervolume Improvement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initial random sampling
        n_init = min(15, self.budget)
        X = lo + (hi - lo) * np.random.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # GP hyperparameters
        alpha = 1e-6  # noise
        length_scales = np.ones(self.dim) * 0.5
        signal_vars = np.ones(self.n_obj)

        def compute_K(X1, X2, length_scales, signal_var):
            """Compute GP kernel matrix with RBF kernel."""
            X1s = X1 / length_scales
            X2s = X2 / length_scales
            sq_dists = np.sum(X1s ** 2, axis=1)[:, None] + np.sum(X2s ** 2, axis=1)[None, :] - 2 * X1s @ X2s.T
            sq_dists = np.maximum(sq_dists, 0)
            K = signal_var ** 2 * np.exp(-0.5 * sq_dists)
            return K

        def predict_gp(X_new, X_train, y_train, length_scales, signal_var, alpha):
            """GP prediction: mean and std for each objective."""
            K = compute_K(X_train, X_train, length_scales, signal_var)
            K[np.diag_indices_from(K)] += alpha
            K_inv_y = np.linalg.solve(K, y_train)
            K_s = compute_K(X_new, X_train, length_scales, signal_var)
            mean = K_s @ K_inv_y
            v = np.linalg.solve(K, K_s.T)
            K_ss = compute_K(X_new, X_new, length_scales, signal_var)
            var = K_ss[np.diag_indices_from(K_ss)] - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mean, std

        def get_pareto_front(F):
            """Get indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            if n == 1:
                return np.array([0])
            
            # A point i is dominated by j if F[j] <= F[i] and F[j] != F[i]
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dominated[i] = True
                            break
            return np.where(~is_dominated)[0]

        def compute_hypervolume(points, ref_point):
            """Compute exact hypervolume in 3D."""
            if len(points) == 0:
                return 0.0
            # Filter points that are better than ref in all objectives
            valid = np.all(points < ref_point, axis=1)
            points = points[valid]
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(points[:, 0])
            points = points[order]
            
            # For each point, compute the volume of the box it contributes
            # A point contributes if it's on the Pareto front
            pareto_idx = get_pareto_front(points)
            if len(pareto_idx) == 0:
                return 0.0
            
            hv = 0.0
            for i in pareto_idx:
                p = points[i]
                # The box from p to ref_point, but we need to subtract overlaps
                # In 3D, we can compute this by considering the "slice"
                # For point i, the contribution is the volume of the box
                # [p[0], next_p[0]] x [p[1], ref[1]] x [p[2], ref[2]]
                # where next_p is the next point in sorted order (or ref)
                # But this is only valid if the points form a proper Pareto front
                
                # More robust: for each point, compute the volume of the box
                # from the point to ref, minus the volumes of boxes from points
                # that dominate it in the other dimensions
                
                # Simple approach: for each point, the contribution is
                # the volume of the box from the point to ref, but only the part
                # not covered by other points
                # This is complex, so let's use a different approach
                
                # For 3D, we can sort by one objective and sweep
                pass
            
            # Alternative: use the formula for 3D hypervolume
            # Sort by x1
            # For each point i, the contribution is:
            # (x1_{i+1} - x1_i) * (ref_x2 - x2_i) * (ref_x3 - x3_i)
            # but only if x2_i and x3_i are "minimal" in some sense
            
            # Let's use a simpler but correct approach for 3D:
            # For each point, compute the volume of the box from the point to ref
            # and subtract the volumes of boxes from points that are better
            # in at least one objective
            
            # Actually, the standard approach for 3D:
            # 1. Sort points by x1
            # 2. For each point i, find the maximum x2 and x3 among points
            #    with x1 < x1_i (i.e., points that are better in x1)
            # 3. The contribution of point i is:
            #    (min(x1_{i+1}, ref_x1) - x1_i) * 
            #    (ref_x2 - max(x2_i, max_x2_prev)) * 
            #    (ref_x3 - max(x3_i, max_x3_prev))
            # where max_x2_prev and max_x3_prev are the maximum x2 and x3
            # among points with x1 < x1_i
            
            # Let me implement this properly
            n_pts = len(points)
            # Sort by x1
            order = np.argsort(points[:, 0])
            pts = points[order]
            
            hv = 0.0
            max_x2 = 0.0
            max_x3 = 0.0
            
            for i in range(n_pts):
                p = pts[i]
                x1 = p[0]
                x2 = p[1]
                x3 = p[2]
                
                # The next x1 (or ref_x1 if this is the last point)
                if i < n_pts - 1:
                    next_x1 = pts[i + 1][0]
                else:
                    next_x1 = ref_point[0]
                
                # The effective x2 and x3 are the max of current and previous max
                eff_x2 = max(x2, max_x2)
                eff_x3 = max(x3, max_x3)
                
                # Only contribute if the box has positive volume
                if x1 < next_x1 and eff_x2 < ref_point[1] and eff_x3 < ref_point[2]:
                    vol = (next_x1 - x1) * (ref_point[1] - eff_x2) * (ref_point[2] - eff_x3)
                    hv += vol
                
                # Update max_x2 and max_x3
                max_x2 = max(max_x2, x2)
                max_x3 = max(max_x3, x3)
            
            return hv

        def ehvi_candidate(X_new, X_train, F_train, ref_point, length_scales, signal_vars, alpha, pareto_indices):
            """Compute EHVI for a candidate point."""
            # Predict each objective
            means = []
            stds = []
            for j in range(self.n_obj):
                y_j = F_train[:, j]
                mu, sd = predict_gp(X_new, X_train, y_j, length_scales, signal_vars[j], alpha)
                means.append(mu[0])
                stds.append(sd[0])
            means = np.array(means)
            stds = np.array(stds)
            
            # Only consider points potentially better than ref
            # EHVI approximation via sampling
            n_samples = 100
            rng_local = np.random.RandomState(42)
            samples = means + stds * rng_local.standard_normal((n_samples, self.n_obj))
            
            # Compute hypervolume contribution
            hv_improvements = []
            for s in samples:
                if np.any(s >= ref_point):
                    continue
                
                # Check if this sample is non-dominated by current front
                dominated_by_existing = False
                for i in range(len(F_train)):
                    if np.all(F_train[i] <= s) and np.any(F_train[i] < s):
                        dominated_by_existing = True
                        break
                if dominated_by_existing:
                    hv_improvements.append(0.0)
                    continue
                
                # Compute HV of the set of non-dominated points including s
                # Get current Pareto front
                current_front = F_train[pareto_indices]
                # Add the sample
                new_front = np.vstack([current_front, s])
                new_pareto_idx = get_pareto_front(new_front)
                
                # Compute HV of current front
                hv_current = compute_hypervolume(current_front, ref_point)
                # Compute HV of new front
                hv_new = compute_hypervolume(new_front[new_pareto_idx], ref_point)
                
                hv_improvements.append(max(0.0, hv_new - hv_current))
            
            ehvi_val = np.mean(hv_improvements)
            return ehvi_val

        # Main optimization loop
        while used < self.budget:
            # Generate candidate points
            n_candidates = min(30, self.budget - used)
            X_cand = []
            
            # Random candidates
            n_rand = max(5, n_candidates // 2)
            X_cand.append(lo + (hi - lo) * np.random.random((n_rand, self.dim)))
            
            # Local perturbations around best points
            n_local = n_candidates - n_rand
            if len(X) > 0 and n_local > 0:
                # Pick a few good points to perturb
                n_perturb = min(5, len(X))
                idx_perturb = np.random.choice(len(X), size=n_perturb, replace=False)
                for idx in idx_perturb:
                    n_per_point = max(1, n_local // n_perturb)
                    perturb = X[idx][None, :] + 0.1 * (hi - lo) * np.random.standard_normal((n_per_point, self.dim))
                    perturb = np.clip(perturb, lo, hi)
                    X_cand.append(perturb)
            
            X_cand = np.vstack(X_cand)
            X_cand = np.clip(X_cand, lo, hi)
            
            # Evaluate EHVI for each candidate
            pareto_indices = get_pareto_front(F)
            
            ehvi_vals = []
            for x_c in X_cand:
                x_c = x_c.reshape(1, -1)
                try:
                    val = ehvi_candidate(x_c, X, F, ref_point, length_scales, signal_vars, alpha, pareto_indices)
                except Exception:
                    val = 0.0
                ehvi_vals.append(val)
            
            ehvi_vals = np.array(ehvi_vals)
            best_idx = np.argmax(ehvi_vals)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            used += 1
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            
            # Update hyperparameters occasionally
            if used % 20 == 0:
                # Simple hyperparameter update: adjust length scales based on spread
                data_range = hi - lo
                length_scales = np.maximum(length_scales * 1.1, 0.01)
                length_scales = np.minimum(length_scales, data_range * 0.5)
                
                # Update signal variance to match data variance
                for j in range(self.n_obj):
                    var_y = np.var(F[:, j])
                    if var_y > 1e-6:
                        signal_vars[j] = np.sqrt(var_y)
        
        return X
# EVOLVE-BLOCK-END
