"""Bayesian Optimization with GP surrogates and EHVI acquisition for RE34 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Bayesian multi-objective optimizer using GP surrogates and EHVI acquisition."""

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

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # Storage for all evaluated points
        X_eval = []
        F_eval = []

        def evaluate(x):
            """Evaluate a single point and store it."""
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X_eval.append(x)
            F_eval.append(f)
            return f

        # Initial random sampling
        n_init = min(15, budget)
        rng = np.random.default_rng(42)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # GP-based Bayesian optimization loop
        # We use a simple GP implementation with RBF kernel

        def gp_predict(X_train, y_train, X_test, length_scale=0.5, noise=1e-8):
            """Simple GP prediction with RBF kernel.
            Returns mean and std for X_test."""
            n = len(X_train)
            d = X_train.shape[1]

            # Standardize inputs
            X_train_std = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
            X_test_std = (X_test - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)

            # RBF kernel
            def rbf(K1, K2, ls):
                sq_dists = np.sum(K1**2, axis=1)[:, None] + np.sum(K2**2, axis=1)[None, :] - 2 * K1 @ K2.T
                return np.exp(-0.5 * sq_dists / (ls**2))

            K = rbf(X_train_std, X_train_std, length_scale) + noise * np.eye(n)
            K_star = rbf(X_train_std, X_test_std, length_scale)

            # Handle potential numerical issues
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                v = np.linalg.solve(L, K_star.T)
                mu = v.T @ alpha
                var = 1.0 - np.sum(v**2, axis=0)
                std = np.sqrt(np.maximum(var, 0))
            except np.linalg.LinAlgError:
                # Fallback to pinv
                try:
                    K_inv = np.linalg.pinv(K)
                    alpha = K_inv @ y_train
                    mu = K_star @ alpha
                    var = 1.0 - np.sum(K_star @ K_inv * K_star.T, axis=1)
                    std = np.sqrt(np.maximum(var, 0))
                except:
                    mu = np.full(len(X_test), y_train.mean())
                    std = np.full(len(X_test), y_train.std() + 1e-6)

            return mu, std

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points with given reference point.
            Points must be strictly better than ref in all objectives."""
            points = np.array(points)
            # Filter points that dominate ref
            mask = np.all(points < ref, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0

            # Sort by first objective
            idx = np.argsort(points[:, 0])
            points = points[idx]

            # For 3D, we can use the inclusion-exclusion approach
            # Sort by x, then for each point compute the 2D hypervolume of the region
            # where this point is the minimum in x

            n = len(points)
            if n == 1:
                return float(np.prod(ref - points[0]))

            # Sort by first coordinate
            order = np.argsort(points[:, 0])
            points = points[order]

            # Compute hypervolume using the standard algorithm for 3D
            # We sort by x and sweep
            hv = 0.0
            for i in range(n):
                # Region where point i is the minimum in x
                x_min = points[i, 0]
                x_max = points[i+1, 0] if i + 1 < n else ref[0]

                # For this x-range, we need the 2D hypervolume of points j where points[j,0] >= x_min
                # Actually, we need points that are "active" in this x-slice
                # Points j with points[j,0] <= x_max and we consider their y,z contributions

                # Collect all points with x <= x_max (they could contribute to y,z in this slice)
                # But we need the union of rectangles in y-z plane
                # For simplicity, use a 2D hypervolume calculation on the relevant points

                # Points that are within [x_min, x_max) in x-dimension
                # Their y,z values define rectangles from (y,z) to (ref[1], ref[2])
                active = [points[j, 1:3] for j in range(i, n) if points[j, 0] < x_max]
                if not active:
                    continue

                # 2D hypervolume of rectangles anchored at ref[1], ref[2]
                # Sort by y
                active = sorted(active, key=lambda p: p[0])
                hv_2d = 0.0
                max_z = -np.inf
                for y, z in active:
                    if z > max_z:
                        width = ref[1] - y
                        if width > 0:
                            hv_2d += width * (max_z - z) if max_z > z else 0
                        # Actually let me redo this properly
                    max_z = max(max_z, z)

                # Let me use a cleaner 2D HV computation
                hv_2d = _hypervolume_2d(active, ref[1:3])
                hv += hv_2d * (x_max - x_min)

            return hv

        def _hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points with reference point.
            points: list of (y, z) where we want area of union of rectangles [y, ref[0]] x [z, ref[1]]"""
            points = np.array(points)
            if len(points) == 0:
                return 0.0

            # Sort by y
            idx = np.argsort(points[:, 0])
            points = points[idx]

            hv = 0.0
            max_z = -np.inf
            for i in range(len(points)):
                y, z = points[i]
                if z > max_z:
                    width = ref[0] - y
                    if width > 0 and max_z > -np.inf:
                        hv += width * (max_z - z)
                    elif width > 0:
                        # First point, no previous max_z
                        pass
                max_z = max(max_z, z)

            # Add the last segment
            # Actually, let me use a cleaner approach
            # Sort by y, keep track of max z seen so far from the right
            idx = np.argsort(points[:, 0])
            points = points[idx]
            n = len(points)
            max_z = np.full(n, -np.inf)
            for i in range(n-1, -1, -1):
                max_z[i] = max(points[i, 1], max_z[i+1]) if i+1 < n else points[i, 1]

            hv = 0.0
            for i in range(n):
                if i + 1 < n:
                    y_next = points[i+1, 0]
                else:
                    y_next = ref[0]
                width = y_next - points[i, 0]
                if width > 0:
                    hv += width * max_z[i]

            return hv

        def ehvi_acquisition(X_train, F_train, X_cand, ref, n_samples=200):
            """Compute Expected Hypervolume Improvement for candidate points.
            Uses Monte Carlo sampling from GP posteriors."""
            d = X_train.shape[1]
            m = F_train.shape[1]

            # Standardize training data
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            X_train_std = (X_train - X_mean) / X_std

            # Fit GPs for each objective
            gps = []
            for j in range(m):
                y = F_train[:, j]
                # Standardize y
                y_mean = y.mean()
                y_std = y.std() + 1e-8
                y_std_arr = (y - y_mean) / y_std

                def rbf(K1, K2, ls=0.5):
                    sq_dists = np.sum(K1**2, axis=1)[:, None] + np.sum(K2**2, axis=1)[None, :] - 2 * K1 @ K2.T
                    return np.exp(-0.5 * sq_dists / (ls**2))

                K = rbf(X_train_std, X_train_std) + 1e-8 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_std_arr))
                except:
                    alpha = np.linalg.pinv(K) @ y_std_arr

                gps.append((X_mean, X_std, y_mean, y_std, X_train_std, L, alpha))

            def gp_sample(gps, X_cand_std, n_samples):
                """Sample from GP posteriors at candidate points."""
                n_cand = len(X_cand_std)
                samples = np.zeros((n_samples, n_cand, m))
                for j in range(m):
                    X_mean, X_std, y_mean, y_std, X_train_std, L, alpha = gps[j]
                    K_star = rbf(X_train_std, X_cand_std)
                    v = np.linalg.solve(L, K_star.T)
                    mu = v.T @ alpha
                    var = 1.0 - np.sum(v**2, axis=0)
                    std = np.sqrt(np.maximum(var, 0))
                    # Sample
                    eps = np.random.default_rng(j*1000).standard_normal((n_samples, n_cand))
                    samples[:, :, j] = (mu[None, :] + std[None, :] * eps) * y_std + y_mean
                return samples

            # Baseline HV
            F_baseline = np.array(F_train)
            mask = np.all(F_baseline < ref, axis=1)
            F_base = F_baseline[mask]
            hv_base = hypervolume_3d(F_base, ref) if len(F_base) > 0 else 0.0

            # Evaluate EHVI for each candidate
            n_cand = len(X_cand)
            X_cand_std = (X_cand - X_mean) / X_std

            ehvi = np.zeros(n_cand)
            for i in range(n_cand):
                # Sample from GP at this candidate
                samples = gp_sample(gps, X_cand_std[i:i+1], n_samples)[:, 0, :]  # shape (n_samples, m)

                # For each sample, compute HV including this new point
                total_improvement = 0.0
                for s in range(n_samples):
                    new_point = samples[s]
                    if np.all(new_point < ref):
                        # Compute HV of baseline + new point
                        all_points = np.vstack([F_base, new_point]) if len(F_base) > 0 else new_point.reshape(1, -1)
                        hv_new = hypervolume_3d(all_points, ref)
                    else:
                        hv_new = hv_base
                    total_improvement += max(0, hv_new - hv_base)

                ehvi[i] = total_improvement / n_samples

            return ehvi

        # Main BO loop
        while len(X_eval) < budget:
            # Get current Pareto front
            F_arr = np.array(F_eval)
            X_arr = np.array(X_eval)

            # Find non-dominated points
            n_pts = len(F_arr)
            is_dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break
            pareto_idx = np.where(~is_dominated)[0]
            pareto_X = X_arr[pareto_idx]
            pareto_F = F_arr[pareto_idx]

            if len(pareto_X) < 3:
                # Not enough points, just sample randomly
                x = lo + (hi - lo) * rng.random(dim)
                evaluate(x)
                continue

            # Generate candidate points using a combination of:
            # 1. Points near the Pareto front
            # 2. Random points
            # 3. Points suggested by GP uncertainty

            # First, get GP predictions to identify high-uncertainty regions
            X_mean = X_arr.mean(axis=0)
            X_std = X_arr.std(axis=0) + 1e-8

            # Generate candidates
            n_cand = 50
            candidates = []

            # Some candidates near Pareto points
            for i in range(min(10, len(pareto_X))):
                for _ in range(3):
                    x = pareto_X[i] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                    x = np.clip(x, lo, hi)
                    candidates.append(x)

            # Some random candidates
            for _ in range(20):
                x = lo + (hi - lo) * rng.random(dim)
                candidates.append(x)

            # Some candidates in high-uncertainty regions (based on GP)
            # Use the GP to find points with high predictive variance
            # For simplicity, sample from a Latin hypercube
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            lhc = sampler.random(n=20)
            for x in lhc:
                x = lo + (hi - lo) * x
                candidates.append(x)

            candidates = np.array(candidates[:n_cand])

            # Compute EHVI for each candidate
            try:
                ehvi_vals = ehvi_acquisition(X_arr, F_arr, candidates, ref, n_samples=50)
                best_idx = np.argmax(ehvi_vals)
            except:
                best_idx = rng.integers(len(candidates))

            x_next = candidates[best_idx]
            evaluate(x_next)

        return X_eval
# EVOLVE-BLOCK-END
