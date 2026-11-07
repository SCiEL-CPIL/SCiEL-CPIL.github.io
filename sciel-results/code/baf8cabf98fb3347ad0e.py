"""Multi-objective optimizer on RE34 with surrogate-assisted EHVI search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted multi-objective optimizer using EHVI for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        def is_better_than_ref(f):
            return np.all(f < ref)

        # ---- Hypervolume utilities ----
        def hv_2d(points_2d, ref_2d):
            m = len(points_2d)
            if m == 0:
                return 0.0
            if m == 1:
                return float((ref_2d[0] - points_2d[0, 0]) * (ref_2d[1] - points_2d[0, 1]))
            pts2 = points_2d[np.argsort(points_2d[:, 0])]
            hv = 0.0
            min_y = ref_2d[1]
            for i in range(m - 1, -1, -1):
                p = pts2[i]
                if p[1] < min_y:
                    min_y = p[1]
                x_next = pts2[i + 1, 0] if i + 1 < m else ref_2d[0]
                hv += (x_next - p[0]) * (ref_2d[1] - min_y)
            return hv

        def hv_3d(points, ref_point):
            n = len(points)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref_point - points[0]))
            pts = points[np.argsort(points[:, 0])]
            hv = 0.0
            for i in range(n):
                x_i = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref_point[0]
                yz = pts[:i + 1, 1:3]
                hv += (x_next - x_i) * hv_2d(yz, ref_point[1:3])
            return hv

        def is_pareto(F_arr):
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=bool)
            pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not pareto[i]:
                    continue
                for j in range(n):
                    if i == j or not pareto[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        pareto[i] = False
                        break
            return pareto

        # ---- Gaussian Process surrogate ----
        def rbf_kernel(X1, X2, length_scales):
            """Compute RBF kernel matrix with ARD length scales."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (length_scales ** 2))

        def gp_predict(X_train, y_train, X_test, length_scales, noise_var):
            """Predict mean and variance using GP with RBF kernel."""
            K = rbf_kernel(X_train, X_train, length_scales) + noise_var * np.eye(len(X_train))
            K_inv = np.linalg.inv(K)
            K_s = rbf_kernel(X_test, X_train, length_scales)
            mean = K_s @ K_inv @ y_train
            K_ss = rbf_kernel(X_test, X_test, length_scales)
            var = np.diag(K_ss) - np.sum(K_s @ K_inv * K_s.T, axis=1)
            var = np.maximum(var, 1e-10)
            return mean, var

        # ---- EHVI computation ----
        def compute_ehvi(candidate, pred_mean, pred_var, pareto_F, ref_point, n_samples=50):
            """Compute Expected HV Improvement for a candidate point."""
            if len(pareto_F) == 0:
                # EHVI = E[max(0, HV({y}) - HV(empty))] = E[max(0, prod(ref - y))]
                samples = pred_mean + np.sqrt(pred_var) * np.random.randn(n_samples, n_obj)
                samples = np.maximum(samples, 0)  # objectives are positive
                gains = []
                for s in samples:
                    if np.all(s < ref_point):
                        gains.append(np.prod(ref_point - s))
                    else:
                        gains.append(0.0)
                return np.mean(gains)
            
            # For non-empty Pareto set, approximate EHVI
            # Use Monte Carlo: sample y ~ N(pred_mean, pred_var), compute HV gain
            rng_local = np.random.RandomState(12345)
            samples = pred_mean + np.sqrt(pred_var) * rng_local.randn(n_samples, n_obj)
            gains = []
            for s in samples:
                if not np.all(s < ref_point):
                    gains.append(0.0)
                    continue
                # Compute HV with and without this point
                with_pt = np.vstack([pareto_F, [s]])
                mask_with = np.all(with_pt < ref_point, axis=1)
                with_pt = with_pt[mask_with]
                hv_with = hv_3d(with_pt, ref_point) if len(with_pt) > 0 else 0.0
                mask_without = np.all(pareto_F < ref_point, axis=1)
                without_pt = pareto_F[mask_without]
                hv_without = hv_3d(without_pt, ref_point) if len(without_pt) > 0 else 0.0
                gain = max(0.0, hv_with - hv_without)
                gains.append(gain)
            return np.mean(gains)

        # ---- Initialize with Latin Hypercube Sampling ----
        rng = np.random.RandomState(42)
        pop_size = min(30, budget)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Filter valid points
        valid_mask = np.array([is_better_than_ref(f) for f in F])
        pareto_X = X[valid_mask].copy()
        pareto_F = F[valid_mask].copy()

        # GP hyperparameters
        length_scales = np.ones(dim) * 0.5
        noise_var = 1e-4

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get current Pareto set
            if len(pareto_F) > 0:
                p_mask = is_pareto(pareto_F)
                pareto_F_cur = pareto_F[p_mask]
                pareto_X_cur = pareto_X[p_mask]
            else:
                pareto_F_cur = pareto_F
                pareto_X_cur = pareto_X

            # Generate candidate points: mix of local and global
            n_candidates = 20
            candidates = np.zeros((n_candidates, dim))
            
            # Half from local perturbation around best HV-contributing point
            if len(pareto_F_cur) > 0:
                # Compute HV contributions for selection
                n_pf = len(pareto_F_cur)
                contribs = np.zeros(n_pf)
                for i in range(n_pf):
                    others = np.delete(pareto_F_cur, i, axis=0)
                    if len(others) == 0:
                        contribs[i] = float(np.prod(ref - pareto_F_cur[i]))
                    else:
                        with_pt = np.vstack([others, [pareto_F_cur[i]]])
                        mask_with = np.all(with_pt < ref, axis=1)
                        with_pt = with_pt[mask_with]
                        hv_with = hv_3d(with_pt, ref) if len(with_pt) > 0 else 0.0
                        mask_without = np.all(others < ref, axis=1)
                        without_pt = others[mask_without]
                        hv_without = hv_3d(without_pt, ref) if len(without_pt) > 0 else 0.0
                        contribs[i] = max(0.0, hv_with - hv_without)
                
                contribs = np.clip(contribs, 1e-10, None)
                probs = contribs / contribs.sum()
                
                # Sample parents
                n_local = n_candidates // 2
                parent_indices = rng.choice(n_pf, size=n_local, replace=True, p=probs)
                for k, pi in enumerate(parent_indices):
                    parent = pareto_X_cur[pi]
                    step = 0.2 * (hi - lo)
                    candidates[k] = parent + step * rng.standard_normal(dim)
                    candidates[k] = np.clip(candidates[k], lo, hi)
            
            # Rest from global random
            n_global = n_candidates - (n_local if len(pareto_F_cur) > 0 else 0)
            if n_global > 0:
                candidates[n_local:] = lo + rng.random((n_global, dim)) * (hi - lo)

            # Predict using GP
            if len(pareto_F_cur) > 5:
                X_train = pareto_X_cur
                y_train = pareto_F_cur
                pred_means, pred_vars = gp_predict(X_train, y_train, candidates, length_scales, noise_var)
            else:
                # Not enough data for GP, use random
                pred_means = np.zeros((n_candidates, n_obj))
                pred_vars = np.ones((n_candidates, n_obj)) * 10.0

            # Compute EHVI for each candidate
            ehvi_scores = np.zeros(n_candidates)
            for k in range(n_candidates):
                ehvi_scores[k] = compute_ehvi(
                    candidates[k], pred_means[k], pred_vars[k],
                    pareto_F_cur, ref, n_samples=30
                )

            # Select best candidate
            best_k = np.argmax(ehvi_scores)
            child = candidates[best_k]
            fc = problem(child)
            used += 1

            # Update archives
            if is_better_than_ref(fc):
                pareto_X = np.vstack([pareto_X, [child]])
                pareto_F = np.vstack([pareto_F, [fc]])

            # Prune Pareto set periodically
            if len(pareto_F) > 60:
                p_mask = is_pareto(pareto_F)
                pareto_X = pareto_X[p_mask]
                pareto_F = pareto_F[p_mask]
                if len(pareto_F) > 50:
                    # Keep top 50 by HV contribution
                    n_pf = len(pareto_F)
                    contribs = np.zeros(n_pf)
                    for i in range(n_pf):
                        others = np.delete(pareto_F, i, axis=0)
                        if len(others) == 0:
                            contribs[i] = float(np.prod(ref - pareto_F[i]))
                        else:
                            with_pt = np.vstack([others, [pareto_F[i]]])
                            mask_with = np.all(with_pt < ref, axis=1)
                            with_pt = with_pt[mask_with]
                            hv_with = hv_3d(with_pt, ref) if len(with_pt) > 0 else 0.0
                            mask_without = np.all(others < ref, axis=1)
                            without_pt = others[mask_without]
                            hv_without = hv_3d(without_pt, ref) if len(without_pt) > 0 else 0.0
                            contribs[i] = max(0.0, hv_with - hv_without)
                    top_idx = np.argsort(contribs)[-50:]
                    pareto_X = pareto_X[top_idx]
                    pareto_F = pareto_F[top_idx]

        return X

# EVOLVE-BLOCK-END
