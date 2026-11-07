"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and EHVI acquisition for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_hypervolume(self, points, ref):
        """Compute exact hypervolume in 3D using inclusion-exclusion."""
        if len(points) == 0:
            return 0.0
        # Filter points that are strictly better than ref in all objectives
        valid = [p for p in points if all(p[i] < ref[i] for i in range(self.n_obj))]
        if len(valid) == 0:
            return 0.0

        # Remove dominated points (keep only non-dominated)
        nondom = []
        for i, p in enumerate(valid):
            dominated = False
            for j, q in enumerate(valid):
                if i != j and all(q[k] <= p[k] for k in range(self.n_obj)) and any(q[k] < p[k] for k in range(self.n_obj)):
                    dominated = True
                    break
            if not dominated:
                nondom.append(p)

        if len(nondom) == 0:
            return 0.0

        # 3D hypervolume calculation using sweep line
        # Sort by first objective
        nondom = sorted(nondom, key=lambda p: p[0])

        # Use a simpler approach: divide into cells
        # For small sets, we can use the standard algorithm
        n = len(nondom)
        if n == 1:
            return max(0, ref[0] - nondom[0][0]) * max(0, ref[1] - nondom[0][1]) * max(0, ref[2] - nondom[0][2])

        # Use inclusion-exclusion for 3D
        # Actually, let's use a more robust approach
        # Sort by x, then compute 2D hypervolume for each slice

        total_hv = 0.0
        for i in range(n):
            x_min = nondom[i][0]
            x_max = nondom[i - 1][0] if i > 0 else ref[0]
            if x_max <= x_min:
                continue
            width = x_max - x_min

            # Points that are "active" in this x-slice: those with x <= x_min (i.e., index <= i)
            # Actually, points with objective[0] <= x_min contribute to the 2D hypervolume in y-z
            active = [p[1:] for p in nondom[:i + 1]]
            if len(active) == 0:
                continue

            # Compute 2D hypervolume of active points with reference [ref[1], ref[2]]
            # Filter to points strictly better than ref in both y and z
            active_valid = [p for p in active if p[0] < ref[1] and p[1] < ref[2]]
            if len(active_valid) == 0:
                continue

            # 2D hypervolume: sort by y, sweep
            active_valid = sorted(active_valid, key=lambda p: p[0])
            hv2d = 0.0
            prev_z = ref[2]
            for p in active_valid:
                y = p[0]
                z = p[1]
                if z < prev_z:
                    hv2d += (ref[1] - y) * (prev_z - z)
                    prev_z = z

            total_hv += width * hv2d

        return total_hv

    def _gp_predict(self, X_train, Y_train, X_test, noise=1e-6):
        """Simple GP prediction using matrix operations."""
        n_train = len(X_train)
        n_test = len(X_test)
        dim = self.dim

        # Normalize inputs
        X_train_min = X_train.min(axis=0)
        X_train_max = X_train.max(axis=0)
        X_range = X_train_max - X_train_min
        X_range[X_range < 1e-10] = 1.0
        X_train_norm = (X_train - X_train_min) / X_range
        X_test_norm = (X_test - X_train_min) / X_range

        # Normalize outputs
        Y_mean = Y_train.mean(axis=0)
        Y_std = Y_train.std(axis=0)
        Y_std[Y_std < 1e-10] = 1.0
        Y_train_norm = (Y_train - Y_mean) / Y_std

        # RBF kernel
        def rbf_kernel(A, B, length_scale=0.5):
            A2 = np.sum(A ** 2, axis=1, keepdims=True)
            B2 = np.sum(B ** 2, axis=1, keepdims=True).T
            dist2 = A2 + B2 - 2 * A @ B.T
            dist2 = np.maximum(dist2, 0)
            return np.exp(-dist2 / (2 * length_scale ** 2))

        # Build kernel matrix
        K = rbf_kernel(X_train_norm, X_train_norm, length_scale=0.5) + noise * np.eye(n_train)

        # Solve for weights
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n_train)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to random prediction
                return Y_train.mean(axis=0) + np.random.randn(n_test, self.n_obj) * Y_train.std(axis=0), \
                       np.ones((n_test, self.n_obj)) * Y_train.std(axis=0)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))

        # Predict
        K_s = rbf_kernel(X_test_norm, X_train_norm, length_scale=0.5)
        mean_norm = K_s @ alpha
        mean = mean_norm * Y_std + Y_mean

        # Variance (approximate)
        v = np.linalg.solve(L, K_s.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 0.01)
        std = np.sqrt(var) * Y_std

        return mean, std

    def _ehvi_acquisition(self, X_candidates, F_train, ref):
        """Compute EHVI acquisition values for candidate points."""
        n_cand = len(X_candidates)
        mean, std = self._gp_predict(F_train[:, :self.dim], F_train[:, self.dim:], X_candidates)

        # Compute current hypervolume
        current_hv = self._compute_hypervolume(F_train[:, self.dim:], ref)

        ehvi = np.zeros(n_cand)
        n_samples = 50

        for i in range(n_cand):
            ehvi_sum = 0.0
            for s in range(n_samples):
                # Sample from GP posterior
                f_sample = mean[i] + std[i] * np.random.randn(self.n_obj)
                f_sample = np.maximum(f_sample, 0)  # Objectives are non-negative

                # Check if strictly better than ref
                if not all(f_sample[j] < ref[j] for j in range(self.n_obj)):
                    continue

                # Compute new hypervolume
                new_points = list(F_train[:, self.dim:]) + [f_sample]
                new_hv = self._compute_hypervolume(new_points, ref)
                improvement = max(0, new_hv - current_hv)
                ehvi_sum += improvement

            ehvi[i] = ehvi_sum / n_samples

        return ehvi

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([3.0, 5.0, 7.0])

        # Initial random sampling
        n_init = 30
        X_init = lo + range_ * np.random.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        eval_count = n_init

        # Store all evaluated points
        X_all = X_init.copy()
        F_all = F_init.copy()

        # Archive of non-dominated points
        archive_X = X_init.copy()
        archive_F = F_init.copy()

        def update_archive(X_new, F_new):
            nonlocal archive_X, archive_F
            new_archive_X = []
            new_archive_F = []
            for x, f in zip(X_new, F_new):
                # Check if f is dominated by any archive point
                dominated = False
                for f_arch in archive_F:
                    if np.all(f_arch <= f) and np.any(f_arch < f):
                        dominated = True
                        break
                if dominated:
                    continue
                # Remove archive points dominated by f
                new_archive_X_temp = []
                new_archive_F_temp = []
                for x_arch, f_arch in zip(archive_X, archive_F):
                    if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                        new_archive_X_temp.append(x_arch)
                        new_archive_F_temp.append(f_arch)
                archive_X = np.array(new_archive_X_temp) if new_archive_X_temp else np.empty((0, dim))
                archive_F = np.array(new_archive_F_temp) if new_archive_F_temp else np.empty((0, n_obj))
                archive_X = np.vstack([archive_X, x.reshape(1, dim)])
                archive_F = np.vstack([archive_F, f.reshape(1, n_obj)])

        # Bayesian optimization loop
        while eval_count < self.budget:
            if time.time() > deadline:
                break

            # Generate candidate points
            n_candidates = 200
            X_cand = lo + range_ * np.random.random((n_candidates, dim))

            # Also add perturbations of current best points
            if len(archive_X) > 0:
                for x_arch in archive_X[:10]:
                    for scale in [0.01, 0.05, 0.1]:
                        x_pert = x_arch + scale * range_ * np.random.randn(dim)
                        x_pert = np.clip(x_pert, lo, hi)
                        X_cand = np.vstack([X_cand, x_pert.reshape(1, dim)])

            # Compute EHVI acquisition
            # Build training data: [X, F] combined
            X_train = X_all
            Y_train = np.hstack([X_all, F_all])

            ehvi = self._ehvi_acquisition(X_cand, Y_train, ref)

            # Select top 10 candidates
            n_select = min(10, self.budget - eval_count)
            top_idx = np.argsort(ehvi)[::-1][:n_select]

            for idx in top_idx:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x_new = X_cand[idx]
                f_new = problem(x_new)
                eval_count += 1

                X_all = np.vstack([X_all, x_new.reshape(1, dim)])
                F_all = np.vstack([F_all, f_new.reshape(1, n_obj)])

                update_archive([x_new], [f_new])

        # Return the archive
        if len(archive_X) > 0:
            return archive_X
        else:
            return X_all[:1]
# EVOLVE-BLOCK-END
