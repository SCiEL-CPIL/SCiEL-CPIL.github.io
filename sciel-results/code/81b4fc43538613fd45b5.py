"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Bayesian Optimization with GP surrogate and EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive):
        """Check if point is non-dominated w.r.t. archive."""
        for a in archive:
            if self._dominates(a, point):
                return False
        return True

    def _add_to_archive(self, point, archive):
        """Add point to archive if non-dominated. Return new archive."""
        if not self._is_nondominated(point, archive):
            return archive
        new_archive = []
        for a in archive:
            if not self._dominates(point, a):
                new_archive.append(a)
        new_archive.append(point)
        return new_archive

    def _hypervolume_3d(self, points, ref):
        """Compute hypervolume for 3D points with reference point using slicing."""
        valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
        if len(valid) == 0:
            return 0.0
        if len(valid) == 1:
            p = valid[0]
            return (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
        
        valid.sort(key=lambda x: x[0])
        hv = 0.0
        for i in range(len(valid)):
            p = valid[i]
            x0_lo = p[0]
            x0_hi = valid[i - 1][0] if i > 0 else ref[0]
            if x0_hi <= x0_lo:
                continue
            w = x0_hi - x0_lo
            hv += w * (ref[1] - p[1]) * (ref[2] - p[2])
        
        return max(0.0, hv)

    def _gp_predict(self, X_train, y_train, X_test, length_scale=0.5, noise=1e-3):
        """GP prediction with RBF kernel, vectorized."""
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_obj = y_train.shape[1]
        
        diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        K = np.exp(-0.5 * dist2 / (length_scale ** 2))
        K += noise * np.eye(n_train)
        
        diff_star = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        dist2_star = np.sum(diff_star ** 2, axis=2)
        K_star = np.exp(-0.5 * dist2_star / (length_scale ** 2))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mean = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            var = 1.0 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-6)
            std = np.sqrt(var)[:, np.newaxis] * np.ones((1, n_obj))
        except np.linalg.LinAlgError:
            try:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
                mean = K_star @ alpha
                std = np.ones((n_test, n_obj)) * 0.5
            except:
                mean = np.tile(y_train.mean(axis=0), (n_test, 1))
                std = np.ones((n_test, n_obj)) * 0.5
        
        return mean, std

    def _ehvi_acquisition(self, X_train, y_train, X_cand, ref, n_samples=15):
        """Compute EHVI acquisition values for candidate points."""
        n_cand = X_cand.shape[0]
        n_obj = self.n_obj
        ehvi_vals = np.zeros(n_cand)
        
        mean, std = self._gp_predict(X_train, y_train, X_cand)
        
        archive_points = []
        for f in y_train:
            if all(f[i] < ref[i] for i in range(n_obj)):
                archive_points.append(f)
        
        if len(archive_points) == 0:
            for i in range(n_cand):
                ehvi_vals[i] = -np.sum(mean[i])
            return ehvi_vals
        
        hv_current = self._hypervolume_3d(archive_points, ref)
        
        rng = np.random.default_rng(42)
        for i in range(n_cand):
            total_improvement = 0.0
            for s in range(n_samples):
                sample_f = mean[i] + std[i] * rng.standard_normal(n_obj)
                if any(sample_f[j] >= ref[j] for j in range(n_obj)):
                    continue
                new_points = archive_points + [sample_f]
                hv_new = self._hypervolume_3d(new_points, ref)
                total_improvement += max(0.0, hv_new - hv_current)
            ehvi_vals[i] = total_improvement / n_samples
        
        return ehvi_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])
        
        total_budget = self.budget
        used = 0
        
        X_train = []
        y_train = []
        archive = []
        
        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            f = problem(np.clip(x, lo, hi))
            used += 1
            return f
        
        # Phase 1: Random initialization
        n_init = min(30, total_budget // 7)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            f = evaluate(x)
            if f is None:
                break
            X_train.append(x)
            y_train.append(f)
            archive = self._add_to_archive(f, archive)
        
        # Phase 2: BO with EHVI
        n_bo_iter = total_budget - used
        
        for bo_iter in range(n_bo_iter):
            if used >= total_budget or time.time() > deadline:
                break
            
            n_cand = 100
            X_cand = lo + range_ * rng.random((n_cand, self.dim))
            
            if len(X_train) >= 3:
                X_train_arr = np.array(X_train)
                y_train_arr = np.array(y_train)
                length_scale = max(0.1, 0.5 * (0.97 ** bo_iter))
                ehvi_vals = self._ehvi_acquisition(X_train_arr, y_train_arr, X_cand, ref, n_samples=15)
                best_idx = np.argmax(ehvi_vals)
            else:
                best_idx = rng.integers(n_cand)
            
            x_new = X_cand[best_idx]
            f_new = evaluate(x_new)
            if f_new is None:
                break
            
            X_train.append(x_new)
            y_train.append(f_new)
            archive = self._add_to_archive(f_new, archive)
        
        return archive
# EVOLVE-BLOCK-END
