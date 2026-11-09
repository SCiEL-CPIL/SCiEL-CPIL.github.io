"""Pure Bayesian Optimization with Gaussian Process surrogate for BBOB f16 (Weierstrass)."""
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Pure Bayesian Optimization using a GP surrogate with EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        # Initialize with diverse points
        rng = np.random.default_rng(42)
        X_obs = []
        Y_obs = []

        # Initial design: center + random points + some structured
        init_points = [
            np.zeros(dim),
            np.array([1.0]*dim),
            np.array([-1.0]*dim),
            np.array([2.0]*dim),
            np.array([-2.0]*dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for x in init_points:
            f, x_eval = call_func(x)
            if f is not None:
                X_obs.append(x_eval.copy())
                Y_obs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()

        if n_calls >= budget or check_time():
            return best_x, best_f

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-10

        def matern52_kernel(X1, X2, ls):
            """Matern 5/2 kernel."""
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
            d2 = np.maximum(d2, 0.0)
            r = np.sqrt(5.0 * d2) / ls
            return signal_var * (1.0 + r + r**2 / 3.0) * np.exp(-r)

        def gp_predict(X_train, Y_train, X_pred, ls, sv, nv):
            """Compute GP mean and variance at X_pred."""
            n_train = len(X_train)
            K = matern52_kernel(X_train, X_train, ls) + nv * np.eye(n_train)
            K_star = matern52_kernel(X_pred, X_train, ls)

            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n_train))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                mean = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = sv - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-12)
                return mean, var
            except np.linalg.LinAlgError:
                # Fallback: use last point's value
                return np.full(len(X_pred), Y_train[-1]), np.full(len(X_pred), 1e-6)

        # Adaptive length scale estimation
        def estimate_length_scale(X, Y, rng):
            """Estimate length scale from observed data."""
            if len(X) < 2:
                return 1.0
            # Use median distance between points with similar function values
            n = len(X)
            d2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
            f_diff = np.abs(Y[:, None] - Y[None, :])
            # Consider pairs with small function difference
            mask = (f_diff < 0.1 * (np.max(Y) - np.min(Y) + 1e-10)) & (np.arange(n)[:, None] < np.arange(n)[None, :])
            if np.sum(mask) > 0:
                dists = np.sqrt(d2[mask])
                # Use median of small distances
                sorted_dists = np.sort(dists)
                idx = min(len(sorted_dists) // 4, len(sorted_dists) - 1)
                ls = max(sorted_dists[idx], 0.1)
                return min(ls, 5.0)
            return 1.0

        # Main Bayesian Optimization loop
        while n_calls < budget and not check_time():
            X_arr = np.array(X_obs)
            Y_arr = np.array(Y_obs)
            n_train = len(X_arr)

            # Adapt length scale periodically
            if n_train >= 10 and n_train % 15 == 0:
                length_scale = estimate_length_scale(X_arr, Y_arr, rng)
                length_scale = max(0.3, min(length_scale, 3.0))

            # Generate candidate points
            n_candidates = 200
            # Mix of local (around best) and global (random) candidates
            n_local = n_candidates // 2
            n_global = n_candidates - n_local

            # Local candidates around best point
            local_radius = 0.5 * (1.0 + 0.5 * np.sin(np.pi * n_calls / budget))
            local_cands = best_x + rng.normal(0, local_radius, (n_local, dim))
            local_cands = np.clip(local_cands, -5, 5)

            # Global candidates
            global_cands = rng.uniform(-5, 5, (n_global, dim))

            candidates = np.vstack([local_cands, global_cands])

            # Compute GP predictions
            mean, var = gp_predict(X_arr, Y_arr, candidates, length_scale, signal_var, noise_var)
            std = np.sqrt(var)

            # Expected Improvement acquisition
            f_best = np.min(Y_arr)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (f_best - mean) / std
                z = np.where(std > 1e-10, z, 0)
                EI = std * (z * norm.cdf(z) + norm.pdf(z))
                EI = np.where(std > 1e-10, EI, 0)

            # Select best candidate
            idx = np.argmax(EI)
            x_new = candidates[idx]
            f_new, x_eval = call_func(x_new)
            if f_new is None:
                break

            X_obs.append(x_eval.copy())
            Y_obs.append(f_new)

            if f_new < best_f:
                best_f = f_new
                best_x = x_eval.copy()

            # Early exit if we're doing very well
            if best_f < 1e-6:
                break

        return best_x, best_f
