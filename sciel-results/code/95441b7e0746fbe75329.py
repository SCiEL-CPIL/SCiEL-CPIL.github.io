"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim

        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initial sampling: Latin Hypercube
        n_init = min(30, self.budget // 3)
        rng = np.random.default_rng(42)
        X = self._latin_hypercube(n_init, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])

        used = n_init

        # Normalize objectives for GP (scale to [0,1] range approximately)
        # We'll use the observed min/max for scaling
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fit GP for each objective
            gps = []
            for k in range(n_obj):
                try:
                    gp = self._fit_gp(X, F_norm[:, k])
                    gps.append(gp)
                except Exception:
                    # Fallback: use simple mean if GP fitting fails
                    gps.append(None)

            # Evaluate EHVI acquisition function on candidate points
            # Generate candidate points: local perturbations of current solutions + some random
            candidates = []
            # Local perturbations around Pareto front
            pareto_idx = self._get_pareto_indices(F)
            for idx in pareto_idx[:10]:
                for sigma in [0.05, 0.1, 0.2]:
                    for _ in range(3):
                        cand = X[idx] + sigma * (hi - lo) * rng.standard_normal(dim)
                        cand = np.clip(cand, lo, hi)
                        candidates.append(cand)
            # Random candidates
            for _ in range(20):
                cand = lo + (hi - lo) * rng.random(dim)
                candidates.append(cand)

            candidates = np.array(candidates)
            n_cand = len(candidates)

            # Evaluate EHVI for each candidate
            best_ehvi = -np.inf
            best_cand = None

            # Use a simplified EHVI: expected improvement in hypervolume
            # Approximate by sampling from GP posterior
            for i in range(n_cand):
                if time.time() > deadline:
                    break
                x_cand = candidates[i]

                # Predict mean and std for each objective
                means = np.zeros(n_obj)
                stds = np.zeros(n_obj)
                valid = True
                for k in range(n_obj):
                    if gps[k] is not None:
                        mu, var = gps[k].predict(x_cand.reshape(1, -1))
                        means[k] = mu[0]
                        stds[k] = np.sqrt(max(var[0], 1e-10))
                    else:
                        means[k] = F_norm[:, k].mean()
                        stds[k] = F_norm[:, k].std() + 0.1

                # Sample from posterior to estimate hypervolume improvement
                n_samples = 10
                sample_F = means + stds[:, None] * rng.standard_normal((n_obj, n_samples))
                sample_F = sample_F.T  # (n_samples, n_obj)

                # Convert back to original scale
                sample_F_orig = sample_F * F_range + F_min

                # Compute hypervolume with and without candidate
                # Current Pareto front hypervolume
                current_hv = self._hypervolume(F, ref_point)

                # For each sample, add the sample point and compute HV
                improvements = []
                for s in range(n_samples):
                    F_aug = np.vstack([F, sample_F_orig[s:s+1]])
                    hv_aug = self._hypervolume(F_aug, ref_point)
                    improvements.append(max(0, hv_aug - current_hv))

                ehvi = np.mean(improvements)

                if ehvi > best_ehvi:
                    best_ehvi = ehvi
                    best_cand = x_cand

            if best_cand is None:
                best_cand = lo + (hi - lo) * rng.random(dim)

            # Evaluate the best candidate
            f_new = problem(best_cand)
            f_new_norm = (f_new - F_min) / F_range

            X = np.vstack([X, best_cand])
            F = np.vstack([F, f_new])
            F_norm = np.vstack([F_norm, f_new_norm])
            used += 1

            # Update normalization if needed
            F_min_new = F.min(axis=0)
            F_max_new = F.max(axis=0)
            F_range_new = F_max_new - F_min_new
            F_range_new[F_range_new == 0] = 1.0
            F_min = F_min_new
            F_max = F_max_new
            F_range = F_range_new
            F_norm = (F - F_min) / F_range

        return X

    def _latin_hypercube(self, n, dim, lo, hi, rng):
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        samples = lo + samples * (hi - lo)
        return samples

    def _fit_gp(self, X, y):
        """Fit a simple GP with RBF kernel."""
        try:
            n = len(y)
            # Standardize y
            y_mean = y.mean()
            y_std = y.std() + 1e-10
            y_std = (y - y_mean) / y_std

            # RBF kernel
            def kernel(X1, X2):
                d = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
                return np.exp(-0.5 * d / 0.5 ** 2)

            K = kernel(X, X) + 1e-6 * np.eye(n)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_std))

            def predict(X_new):
                K_s = kernel(X_new, X)
                mu = K_s @ alpha
                v = np.linalg.solve(L, K_s.T)
                var = 1.0 - np.sum(v ** 2, axis=1)
                mu = mu * y_std + y_mean
                var = var * y_std ** 2
                return mu, np.maximum(var, 1e-10)

            class GP:
                def __init__(self, predict_fn):
                    self._predict = predict_fn
                def predict(self, X_new):
                    return self._predict(X_new)

            return GP(predict)
        except Exception:
            return None

    def _get_pareto_indices(self, F):
        """Get indices of non-dominated points."""
        n = len(F)
        pareto = np.zeros(n, dtype=bool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                pareto[i] = True
        return np.where(pareto)[0]

    def _hypervolume(self, F, ref_point):
        """Compute hypervolume indicator for 3 objectives (minimization)."""
        # Filter points that are better than reference in all objectives
        mask = np.all(F < ref_point, axis=1)
        F_filtered = F[mask]
        if len(F_filtered) == 0:
            return 0.0

        # Sort by first objective
        order = np.argsort(F_filtered[:, 0])
        F_sorted = F_filtered[order]

        # Use recursive hypervolume computation for 3D
        return self._hv3d(F_sorted, ref_point)

    def _hv3d(self, points, ref):
        """Compute 3D hypervolume using inclusion-exclusion."""
        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            p = points[0]
            if np.all(p < ref):
                return np.prod(ref - p)
            return 0.0

        # Sort by x-coordinate
        order = np.argsort(points[:, 0])
        points = points[order]

        # Recursive: for each point, compute the hypervolume of the box
        # from the point to the reference, minus overlaps
        # Simple approach: use the standard algorithm
        hv = 0.0
        for i in range(n):
            p = points[i]
            if np.all(p < ref):
                # Volume of box from p to ref
                vol = np.prod(ref - p)
                # Subtract volumes of boxes from other points that overlap
                # This is a simplification - for exact HV we need more care
                # For now, use a conservative estimate
                hv += vol * 0.5  # crude estimate

        # Better: use the standard 3D HV algorithm
        # Sort by x, then for each point, compute the 2D HV of the "slice"
        return self._hv3d_exact(points, ref)

    def _hv3d_exact(self, points, ref):
        """Exact 3D hypervolume computation."""
        n = len(points)
        if n == 0:
            return 0.0

        # Filter dominated points
        points = self._remove_dominated(points)

        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            p = points[0]
            if np.all(p < ref):
                return np.prod(ref - p)
            return 0.0

        # Sort by first coordinate
        order = np.argsort(points[:, 0])
        points = points[order]

        hv = 0.0
        prev_x = ref[0]
        for i in range(n):
            p = points[i]
            if p[0] >= ref[0]:
                break
            if np.all(p < ref):
                # The x-slice width
                dx = prev_x - p[0]
                # 2D HV of the remaining points in the y-z plane
                # Points that are "active" at this x-slice
                active = points[i:]
                active_2d = active[:, 1:]
                ref_2d = ref[1:]
                hv_2d = self._hv2d(active_2d, ref_2d)
                hv += dx * hv_2d
            prev_x = p[0]

        return hv

    def _hv2d(self, points, ref):
        """Exact 2D hypervolume computation."""
        n = len(points)
        if n == 0:
            return 0.0

        points = self._remove_dominated_2d(points)
        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            p = points[0]
            if np.all(p < ref):
                return np.prod(ref - p)
            return 0.0

        # Sort by first coordinate
        order = np.argsort(points[:, 0])
        points = points[order]

        hv = 0.0
        prev_x = ref[0]
        for i in range(n):
            p = points[i]
            if p[0] >= ref[0]:
                break
            if p[1] < ref[1]:
                dx = prev_x - p[0]
                dy = ref[1] - p[1]
                hv += dx * dy
            prev_x = p[0]

        return hv

    def _remove_dominated(self, points):
        """Remove dominated points (minimization)."""
        n = len(points)
        keep = np.zeros(n, dtype=bool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    dominated = True
                    break
            if not dominated:
                keep[i] = True
        return points[keep]

    def _remove_dominated_2d(self, points):
        """Remove dominated points in 2D (minimization)."""
        n = len(points)
        keep = np.zeros(n, dtype=bool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if points[j][0] <= points[i][0] and points[j][1] <= points[i][1] and \
                   (points[j][0] < points[i][0] or points[j][1] < points[i][1]):
                    dominated = True
                    break
            if not dominated:
                keep[i] = True
        return points[keep]
# EVOLVE-BLOCK-END
