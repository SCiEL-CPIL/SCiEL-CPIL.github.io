"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initialize with a diverse set of points
        X = []
        Y = []

        # Add origin and some structured points
        init_points = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            np.zeros(dim) + 0.1 * rng.standard_normal(dim),
        ]

        for p in init_points:
            p = np.clip(p, -5, 5)
            fp = func(p)
            X.append(p)
            Y.append(fp)

        best_x = X[0].copy()
        best_y = Y[0]
        for i in range(1, len(X)):
            if Y[i] < best_y:
                best_y = Y[i]
                best_x = X[i].copy()

        call_count = len(X)

        # GP hyperparameters
        length_scales = np.ones(dim) * 2.0
        sigma_f = 10.0
        sigma_n = 1e-4

        # Update hyperparameters occasionally
        def update_hyperparams(X_arr, Y_arr, rng):
            nonlocal length_scales, sigma_f, sigma_n
            # Normalize Y for better conditioning
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_norm = (Y_arr - y_mean) / y_std

            # Simple heuristic: adjust based on function variance
            sigma_f = max(1.0, y_std * 2.0)
            # Adjust length scales based on spread of points
            for d in range(dim):
                spread = np.ptp(X_arr[:, d])
                if spread > 1e-6:
                    length_scales[d] = max(0.5, min(5.0, spread * 0.3))
            return

        def compute_kern(X1, X2):
            """Compute RBF kernel matrix."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            scaled = sq_dists / (2 * length_scales**2)
            return sigma_f**2 * np.exp(-scaled)

        def gp_predict(X_new):
            """Predict mean and variance at X_new given training data."""
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            K = compute_kern(X_arr, X_arr)
            K += sigma_n**2 * np.eye(len(X_arr))

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter and retry
                K += 1e-6 * np.eye(len(X_arr))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fall back to ridge regression
                    K_reg = K + 1e-4 * np.eye(len(X_arr))
                    K_inv = np.linalg.inv(K_reg)
                    K_s = compute_kern(X_arr, X_new)
                    mu = K_s.T @ K_inv @ Y_arr
            else:
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                K_s = compute_kern(X_arr, X_new)
                mu = K_s.T @ alpha

            # Variance
            K_ss = compute_kern(X_new, X_new)
            K_s_new = compute_kern(X_arr, X_new)
            if 'L' in dir():
                v = np.linalg.solve(L, K_s_new)
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
            else:
                var = np.diag(K_ss) - np.sum(K_s_new.T @ K_inv @ K_s_new, axis=0)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        def acquire_EI(x_eval, mu_best):
            """Expected Improvement acquisition."""
            mu, sigma = gp_predict(x_eval)
            if sigma < 1e-8:
                return 0.0
            z = (mu_best - mu) / sigma
            ei = (mu_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return np.mean(ei)

        # Main loop
        while call_count < self.budget:
            if time.time() > deadline:
                break

            # Update hyperparameters every 50 calls
            if call_count % 50 == 0 and call_count > 10:
                update_hyperparams(np.array(X), np.array(Y), rng)

            # Generate candidate points for acquisition
            n_candidates = 200
            # Mix of random and local candidates
            candidates = []

            # Random candidates
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, dim))

            # Local candidates around best point
            for _ in range(n_candidates // 4):
                candidates.append(best_x + 0.5 * rng.standard_normal(dim))

            # Local candidates around recent good points
            if len(X) > 5:
                # Sort by Y and take top 5
                sorted_idx = np.argsort(Y)
                top_idx = sorted_idx[:5]
                for idx in top_idx:
                    candidates.append(X[idx] + 0.3 * rng.standard_normal(dim))

            # Add some structured exploration
            for _ in range(n_candidates // 4):
                base = X[rng.integers(0, len(X))]
                candidates.append(base + 1.0 * rng.standard_normal(dim))

            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5, 5)

            # Evaluate acquisition function
            mu, sigma = gp_predict(candidates)
            z = (best_y - mu) / (sigma + 1e-8)
            EI = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)

            # Pick the best candidate
            best_idx = np.argmax(EI)
            x_new = candidates[best_idx]
            x_new = np.clip(x_new, -5, 5)

            # Evaluate
            f_new = func(x_new)
            call_count += 1

            X.append(x_new)
            Y.append(f_new)

            if f_new < best_y:
                best_y = f_new
                best_x = x_new.copy()

            # Local refinement with Nelder-Mead every 30 calls
            if call_count % 30 == 0 and call_count < self.budget - 10:
                try:
                    res = minimize(func, best_x, method='Nelder-Mead',
                                   options={'maxiter': 5, 'xatol': 1e-3, 'fatol': 1e-4})
                    if res.fun < best_y:
                        best_y = res.fun
                        best_x = res.x.copy()
                        X.append(best_x)
                        Y.append(best_y)
                        call_count += 1
                except Exception:
                    pass

            # Periodic global restart if stuck
            if call_count % 100 == 0:
                x_restart = rng.uniform(-4, 4, dim)
                f_restart = func(x_restart)
                call_count += 1
                X.append(x_restart)
                Y.append(f_restart)
                if f_restart < best_y:
                    best_y = f_restart
                    best_x = x_restart.copy()

        return best_x, best_y

# EVOLVE-BLOCK-END
