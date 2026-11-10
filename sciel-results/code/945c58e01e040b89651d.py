"""Multi-objective Bayesian optimizer for RE34 using GP surrogate and HV-based EI."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian multi-objective optimizer using GP surrogate and hypervolume-based EI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random.default_rng(0)

        # Initialize with random points
        n_init = min(20, budget)
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Track all evaluated points for hypervolume
        all_X = X.copy()
        all_F = F.copy()

        # GP hyperparameters (will be tuned)
        # We use a simple ARD kernel with noise
        def gp_predict(X_train, Y_train, x_new, length_scales, noise_var, signal_var):
            """Compute GP mean and variance for multiple objectives."""
            n = X_train.shape[0]
            d = X_train.shape[1]
            # Compute kernel matrix
            # Normalize inputs
            Xn = (X_train - lo) / (hi - lo + 1e-10)
            xn = (x_new - lo) / (hi - lo + 1e-10)
            
            # ARD kernel
            inv_ls2 = 1.0 / (length_scales ** 2)
            # K[i,j] = signal_var * exp(-0.5 * sum_d (Xn[i,d]-Xn[j,d])^2 / ls[d]^2) + noise if i==j
            diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2 * inv_ls2[np.newaxis, np.newaxis, :], axis=2)
            K = signal_var * np.exp(-0.5 * sq_dist) + np.eye(n) * noise_var
            
            # K_star: kernel between train and new
            diff_star = Xn - xn[np.newaxis, :]
            sq_dist_star = np.sum(diff_star ** 2 * inv_ls2[np.newaxis, :], axis=1)
            K_star = signal_var * np.exp(-0.5 * sq_dist_star)
            
            # Solve for each objective
            means = np.zeros((n_obj, x_new.shape[0]))
            vars = np.zeros((n_obj, x_new.shape[0]))
            
            # Precompute Cholesky
            try:
                L = np.linalg.cholesky(K)
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Add jitter
                K += np.eye(n) * 1e-6
                L = np.linalg.cholesky(K)
                L_inv = np.linalg.inv(L)
            
            v = np.linalg.solve(L, np.linalg.solve(L.T, Y_train.T))  # (n, n_obj)
            v = v.T  # (n_obj, n)
            
            for j in range(n_obj):
                mu = K_star @ v[j]
                var = signal_var - np.sum(K_star ** 2 * np.linalg.solve(L, K_star[np.newaxis, :]).T, axis=1)
                # More efficient: var = k(x,x) - K_star^T K^{-1} K_star
                K_star_col = K_star[:, np.newaxis]
                alpha = np.linalg.solve(L, K_star_col)
                var = signal_var - np.sum(alpha * K_star_col, axis=1)
                var = np.maximum(var, 1e-10)
                means[j] = mu
                vars[j] = var
            
            return means, vars

        def compute_hypervolume(F_points, ref_point):
            """Compute exact hypervolume for 3D using inclusion-exclusion or sorting."""
            # Filter points that dominate reference
            mask = np.all(F_points < ref_point, axis=1)
            F = F_points[mask]
            if len(F) == 0:
                return 0.0
            
            # For 3D, we can use a sorting-based approach
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            
            # Remove dominated points (keep only non-dominated)
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(i + 1, len(F)):
                    if not nd_mask[j]:
                        continue
                    if np.all(F[i] <= F[j]):
                        nd_mask[j] = False
                    elif np.all(F[j] <= F[i]):
                        nd_mask[i] = False
                        break
            F = F[nd_mask]
            if len(F) == 0:
                return 0.0
            
            # 3D hypervolume: sort by f1, sweep with 2D structure
            # Use the standard algorithm: sort by f1, maintain upper envelope in f2-f3
            n = len(F)
            if n == 1:
                return float(np.prod(ref_point - F[0]))
            
            # Sort by f1 ascending
            order = np.argsort(F[:, 0])
            F = F[order]
            
            # For each point, compute its contribution
            # Contribution of point i: (f1[i+1] - f1[i]) * area in 2D of points dominated by i but not by any point with f1 < f1[i]
            # Actually, let's use a simpler approach for small n
            # Inclusion-exclusion is complex; use the sorting sweep
            
            hv = 0.0
            # Sort by f1
            # For each point, the region it uniquely contributes is bounded by:
            # f1 from F[i,0] to min(F[i+1,0], ref[0])
            # and in f2-f3, the region below F[i] that isn't covered by points with smaller f1
            
            # Simpler: for 3D with small n, use the standard O(n^2) or O(n^3) algorithm
            # Let's use a direct approach: for each point, compute the box it contributes
            
            # Better: use the algorithm from Fonseca & Fleming
            # Sort by first objective
            # Maintain a list of "active" points in the 2D projection
            
            # For simplicity with small n (< 100), use O(n^3) exact method
            # Hypervolume = sum over all non-dominated points of the volume of the box
            # that is dominated by this point but not by any other point with a smaller first objective
            
            # Let me use a different approach: recursive or direct
            # For 3D, we can do:
            # Sort by f1
            # For point i, its contribution in f1 dimension is [F[i,0], min(F[i+1,0], ref[0])]
            # In the f2-f3 plane, we need the area of the region [0, F[i,1]] x [0, F[i,2]]
            # minus the union of regions from points j < i that have f2 <= F[i,1] and f3 <= F[i,2]
            
            # This is getting complex. Let's use a numerical approach or a library-free exact method.
            # For 3D, there's a known O(n log n) algorithm, but let me implement a simpler O(n^2) one.
            
            # Actually, let's just use a simple but correct method:
            # The hypervolume is the volume of the union of boxes [F[i], ref] for all i.
            # For 3D, we can use the inclusion-exclusion principle, but that's 2^n terms.
            # Instead, use the sweep line: sort by f1, and for each "slab" between consecutive f1 values,
            # compute the 2D hypervolume of the active points.
            
            f1_vals = np.append(F[:, 0], ref_point[0])
            # Unique f1 breakpoints
            breakpoints = np.unique(f1_vals)
            
            total_hv = 0.0
            for k in range(len(breakpoints) - 1):
                f1_lo = breakpoints[k]
                f1_hi = breakpoints[k + 1]
                if f1_hi <= f1_lo:
                    continue
                # Points active in this slab: those with F[i,0] <= f1_lo
                active = F[F[:, 0] <= f1_lo]
                if len(active) == 0:
                    continue
                # 2D hypervolume of active points in f2-f3 with reference [ref[1], ref[2]]
                # But we only count up to ref
                active_f23 = active[:, 1:3]
                ref_2d = ref_point[1:3]
                hv_2d = _hypervolume_2d(active_f23, ref_2d)
                total_hv += (f1_hi - f1_lo) * hv_2d
            
            return total_hv

        def _hypervolume_2d(points, ref_2d):
            """Compute 2D hypervolume."""
            # Filter
            mask = np.all(points < ref_2d, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by f2
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            # Remove dominated
            nd = [pts[0]]
            for i in range(1, len(pts)):
                if pts[i, 1] < nd[-1][1]:
                    nd.append(pts[i])
                elif pts[i, 0] < nd[-1][0]:
                    # This shouldn't happen after sorting, but handle
                    nd.append(pts[i])
            nd = np.array(nd)
            # Compute hypervolume
            hv = 0.0
            for i in range(len(nd)):
                f2_hi = nd[i + 1, 0] if i + 1 < len(nd) else ref_2d[0]
                f3_hi = ref_2d[1]
                if i + 1 < len(nd) and nd[i + 1, 1] < nd[i, 1]:
                    f3_hi = min(nd[i, 1], ref_2d[1])
                    # Actually, for 2D: the region contributed by point i is
                    # [nd[i,0], nd[i+1,0]] x [0, nd[i,1]] if no other point covers it
                    # After removing dominated and sorting by f2 ascending, f3 descending:
                    # Point i contributes: width * height where
                    # width = (next_f2 - current_f2) if next exists, else (ref_f2 - current_f2)
                    # height = current_f3
                    # But we need to be careful: the height is limited by the next point's f3? No.
                    # In 2D, after sorting by f2 asc and removing dominated (keeping f3 desc),
                    # point i contributes: (f2_{i+1} - f2_i) * f3_i, where f2_{n} = ref_f2
                    pass
            # Let me redo this properly
            # Sort by f2 ascending
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            # Remove dominated: keep points where f3 is strictly decreasing
            nd = [pts[0]]
            for i in range(1, len(pts)):
                if pts[i, 1] < nd[-1][1]:
                    nd.append(pts[i])
            nd = np.array(nd)
            hv = 0.0
            for i in range(len(nd)):
                f2_next = nd[i + 1, 0] if i + 1 < len(nd) else ref_2d[0]
                width = f2_next - nd[i, 0]
                height = nd[i, 1]
                hv += width * height
            return hv

        def hv_contribution(F_new, F_archive, ref_point):
            """Compute the hypervolume contribution of a new point to the archive."""
            # HV with new point minus HV without
            F_combined = np.vstack([F_archive, F_new[np.newaxis, :]])
            hv_with = compute_hypervolume(F_combined, ref_point)
            hv_without = compute_hypervolume(F_archive, ref_point)
            return hv_with - hv_without

        # Initialize GP hyperparameters
        # Use normalized space [0,1]^dim
        length_scales = np.ones(dim) * 0.5
        signal_var = 1.0
        noise_var = 0.01

        # Precompute for speed: normalize inputs
        def normalize(x):
            return (x - lo) / (hi - lo + 1e-10)

        def denormalize(xn):
            return lo + xn * (hi - lo)

        # Main BO loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Get current non-dominated front
            F_current = all_F
            # Find non-dominated points
            nd_mask = np.ones(len(F_current), dtype=bool)
            for i in range(len(F_current)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F_current)):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if np.all(F_current[i] <= F_current[j]) and np.any(F_current[i] < F_current[j]):
                        nd_mask[j] = False
            nd_F = F_current[nd_mask]
            
            if len(nd_F) == 0:
                nd_F = F_current
            
            # Compute current hypervolume
            hv_current = compute_hypervolume(F_current, ref)
            
            # Generate candidate points
            n_candidates = 50
            X_cand = lo + (hi - lo) * rng.random((n_candidates, dim))
            
            # Also add some perturbed points from non-dominated front
            n_perturb = min(10, len(nd_F))
            if n_perturb > 0:
                idx_perturb = rng.integers(0, len(nd_F), size=n_perturb)
                X_perturb = all_X[nd_mask][idx_perturb] + 0.1 * (hi - lo) * rng.standard_normal((n_perturb, dim))
                X_perturb = np.clip(X_perturb, lo, hi)
                X_cand = np.vstack([X_cand, X_perturb])
            
            # Predict with GP
            X_train_norm = normalize(all_X)
            Y_train = all_F
            X_cand_norm = normalize(X_cand)
            
            means, vars = gp_predict(X_train_norm, Y_train, X_cand_norm, length_scales, noise_var, signal_var)
            
            # Compute EI based on hypervolume contribution
            # For each candidate, estimate the expected HV contribution
            # EI_hv = E[max(0, HV_contribution(f))]
            # Approximate: for each objective, EI = max(0, f_best - f_pred) * probability
            # Better: use the GP predictions to estimate the distribution of f, then
            # estimate the expected HV contribution
            
            # Simple approach: for each candidate, compute the mean prediction and use
            # a Monte Carlo estimate of the HV contribution
            n_mc = 20
            ei_scores = np.zeros(len(X_cand))
            
            for i in range(len(X_cand)):
                # Sample from the posterior
                samples = means[:, i] + np.sqrt(vars[:, i]) * rng.standard_normal((n_mc, n_obj))
                # Clip to reasonable range
                samples = np.clip(samples, 0, None)
                
                # For each sample, compute HV contribution
                contributions = []
                for s in samples:
                    # Quick HV contribution: compare to current front
                    # If s is dominated by any point in nd_F, contribution is 0
                    dominated = False
                    for f_nd in nd_F:
                        if np.all(f_nd <= s):
                            dominated = True
                            break
                    if dominated:
                        contributions.append(0.0)
                    else:
                        # Estimate contribution: volume of box dominated by s but not by any nd point
                        # This is complex; approximate by the volume of the box [s, ref]
                        # minus overlaps
                        # Simple approximation: if s improves any objective, estimate contribution
                        # as the product of improvements, clipped
                        improvements = np.maximum(0, nd_F.min(axis=0) - s)
                        # But this overestimates; use a simpler heuristic
                        # Contribution ~ min of improvements across objectives where improvement exists
                        pos_improvements = improvements[improvements > 0]
                        if len(pos_improvements) == 0:
                            contributions.append(0.0)
                        else:
                            # Geometric mean of improvements, scaled
                            contrib = np.prod(pos_improvements) ** (1.0 / len(pos_improvements))
                            contributions.append(contrib)
                
                ei_scores[i] = np.mean(contributions)
            
            # Select best candidate
            best_idx = np.argmax(ei_scores)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = np.array(problem(x_new), dtype=np.float64)
            used += 1
            
            # Update archives
            all_X = np.vstack([all_X, x_new[np.newaxis, :]])
            all_F = np.vstack([all_F, f_new[np.newaxis, :]])
            
            # Update GP hyperparameters every 10 iterations (simple: fixed for now)
            # Could do MLE, but for speed, keep fixed or do simple adaptation
        
        return all_X

# EVOLVE-BLOCK-END
