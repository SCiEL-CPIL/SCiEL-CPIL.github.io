# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


class Optimizer:
    """Bayesian Optimization for multi-objective WFG7 using GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        ref_point = np.array([3.0, 5.0, 7.0])

        # Initial sampling: Latin Hypercube Sampling
        n_init = min(20, budget)
        X_samples = []
        F_samples = []

        for i in range(n_init):
            x = np.array([lo[j] + (hi[j] - lo[j]) * ((i + rng.random()) / n_init) for j in range(dim)])
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_samples.append(x)
            F_samples.append(f)

        X_samples = np.array(X_samples)
        F_samples = np.array(F_samples)
        used = n_init

        def hypervolume_3d(points, ref):
            """Exact 3D hypervolume computation using sweep line."""
            if len(points) == 0:
                return 0.0
            # Filter points better than ref in all objectives
            valid = points[np.all(points < ref, axis=1)]
            if len(valid) == 0:
                return 0.0
            n = len(valid)
            if n == 1:
                return float(np.prod(ref - valid[0]))

            # Sort by first objective
            idx = np.argsort(valid[:, 0])
            points = valid[idx]

            # For 3D, use an O(n^2 log n) algorithm
            # Sort by f1, then for each point compute the area in f2-f3 not covered by later points
            hv = 0.0
            for i in range(n):
                x1, x2, x3 = points[i]
                # Width in f1 direction: from x1 to min(ref[0], next point's f1)
                if i < n - 1:
                    f1_width = points[i + 1, 0] - x1
                else:
                    f1_width = ref[0] - x1

                # Area in f2-f3 plane: rectangle [x2, ref[1]) x [x3, ref[2])
                # minus areas covered by points j > i that have f2 < ref[1] and f3 < ref[2]
                # and dominate in the f2-f3 region
                area = (ref[1] - x2) * (ref[2] - x3)
                # Subtract overlaps with later points
                for j in range(i + 1, n):
                    y2, y3 = points[j, 1], points[j, 2]
                    if y2 < ref[1] and y3 < ref[2]:
                        # Overlap in f2-f3
                        overlap_w = max(0, min(ref[1], y2) - max(x2, y2))
                        overlap_h = max(0, min(ref[2], y3) - max(x3, y3))
                        area -= overlap_w * overlap_h

                hv += f1_width * max(0, area)

            return hv

        def ehvi_acquisition(x, X_train, F_train, ref, length_scales, noise_var=1e-4, n_mc=50):
            """Compute EHVI for candidate x using Monte Carlo approximation."""
            # Build GP model for each objective
            # Normalize X to [0,1] for GP
            X_norm = (X_train - lo) / (hi - lo + 1e-10)
            x_norm = (x - lo) / (hi - lo + 1e-10)

            # GP kernel: Matern 5/2
            def kernel(a, b, ls):
                # a: (n, dim), b: (m, dim)
                diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                sqrt5_dist = np.sqrt(5.0) * dist / ls
                K = (1 + sqrt5_dist + 5 * dist ** 2 / (3 * ls ** 2)) * np.exp(-sqrt5_dist)
                return K

            # Predictive distribution for each objective
            preds = []
            for k in range(n_obj):
                y = F_train[:, k]
                y_mean = np.mean(y)
                y_std = np.std(y) + 1e-10
                y_norm = (y - y_mean) / y_std

                K = kernel(X_norm, X_norm, length_scales[k]) + noise_var * np.eye(len(X_train))
                Kx = kernel(X_norm, x_norm.reshape(1, -1), length_scales[k]).flatten()

                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_train)))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    mu = Kx @ alpha
                    v = np.linalg.solve(L, Kx)
                    var = 1.0 - v @ v
                    var = max(var, 1e-10)
                    preds.append((y_mean + y_std * mu, y_std * np.sqrt(var)))
                except:
                    preds.append((y_mean, y_std))

            # Monte Carlo estimate of EHVI
            hv_current = hypervolume_3d(F_train, ref)
            improvements = []
            for _ in range(n_mc):
                f_sample = np.array([
                    preds[k][0] + preds[k][1] * rng.standard_normal()
                    for k in range(n_obj)
                ])
                # Clip to reasonable range
                f_sample = np.clip(f_sample, np.min(F_train, axis=0) - 1.0, ref)
                new_set = np.vstack([F_train, [f_sample]])
                hv_new = hypervolume_3d(new_set, ref)
                improvements.append(max(0, hv_new - hv_current))

            return np.mean(improvements)

        # Main optimization loop
        # Use fixed length scales, adaptively adjusted
        length_scales = np.ones(n_obj) * 0.3

        while used < budget:
            # Generate candidate points for acquisition
            candidates = []

            # Strategy 1: Local perturbations around best non-dominated points
            non_dom_mask = self._non_dominated_mask(F_samples)
            X_nd = X_samples[non_dom_mask]
            F_nd = F_samples[non_dom_mask]

            if len(X_nd) > 0:
                for i in range(min(5, len(X_nd))):
                    base = X_nd[i]
                    for trial in range(3):
                        step = 0.15 * (hi - lo)
                        x = base + step * rng.standard_normal(dim)
                        x = np.clip(x, lo, hi)
                        candidates.append(x)

            # Strategy 2: Random exploration
            for _ in range(5):
                x = lo + (hi - lo) * rng.random(dim)
                candidates.append(x)

            # Strategy 3: Interpolation between non-dominated points
            if len(X_nd) >= 2:
                for _ in range(5):
                    i, j = rng.integers(0, len(X_nd), size=2)
                    if i != j:
                        alpha = rng.random()
                        x = alpha * X_nd[i] + (1 - alpha) * X_nd[j]
                        x = np.clip(x, lo, hi)
                        candidates.append(x)

            # Evaluate EHVI for each candidate and select the best
            best_candidate = None
            best_ehvi = -np.inf

            for x in candidates:
                ehvi_val = ehvi_acquisition(
                    x, X_samples, F_samples, ref_point,
                    length_scales, noise_var=1e-4, n_mc=30
                )
                if ehvi_val > best_ehvi:
                    best_ehvi = ehvi_val
                    best_candidate = x

            if best_candidate is None:
                best_candidate = lo + (hi - lo) * rng.random(dim)

            # Evaluate the best candidate
            f_new = np.array(problem(best_candidate))
            used += 1

            X_samples = np.vstack([X_samples, [best_candidate]])
            F_samples = np.vstack([F_samples, [f_new]])

            # Adapt length scales based on observed variance (simple heuristic)
            # Reduce length scales if we're finding new non-dominated points frequently
            if self._is_non_dominated(f_new, F_samples[:-1]):
                length_scales = np.maximum(length_scales * 0.95, 0.05)

        return X_samples

    def _non_dominated_mask(self, F):
        """Return mask of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
        return mask

    def _is_non_dominated(self, f, F):
        """Check if f is non-dominated with respect to set F."""
        for fj in F:
            if np.all(fj <= f) and np.any(fj < f):
                return False
        return True

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2."""
        return np.all(f1 <= f2) and np.any(f1 < f2)
# EVOLVE-BLOCK-END
