"""Bayesian Optimization with GP surrogate and EI acquisition for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian Optimization with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initialize with a small set of points (Latin hypercube-ish)
        n_init = 12
        X = np.zeros((n_init, dim))
        for d in range(dim):
            X[:, d] = rng.uniform(-5, 5, n_init)
        # Add a few more spread points
        X = np.vstack([X, np.zeros((1, dim))])
        X = np.clip(X, -5, 5)
        n_init = X.shape[0]

        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)].copy()
        best_y = float(Y.min())
        calls_made = n_init

        # GP hyperparameters (RBF kernel)
        # We'll use a simple approach: fit length-scales and variance from data
        # For speed, use a fixed reasonable kernel and only update noise

        # Helper: compute GP posterior mean and variance at points X_new given training (X, Y)
        def gp_predict(X_new, X_train, Y_train, length_scales, sigma_f, sigma_n):
            """
            Compute GP posterior mean and std at X_new.
            length_scales: array of shape (dim,)
            sigma_f: signal variance
            sigma_n: noise variance
            """
            n = X_train.shape[0]
            m = X_new.shape[0]
            d = X_train.shape[1]

            # Squared exponential kernel
            # K(x, x') = sigma_f^2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / l_d^2)
            diff = X_train[:, None, :] - X_new[None, :, :]
            sq_dist = np.sum(diff ** 2 / length_scales[None, None, :] ** 2, axis=2)
            K = sigma_f ** 2 * np.exp(-0.5 * sq_dist)

            # Training kernel matrix
            diff_t = X_train[:, None, :] - X_train[None, :, :]
            sq_dist_t = np.sum(diff_t ** 2 / length_scales[None, None, :] ** 2, axis=2)
            K_t = sigma_f ** 2 * np.exp(-0.5 * sq_dist_t)
            K_t[np.diag_indices_from(K_t)] += sigma_n ** 2

            # Solve
            L = np.linalg.cholesky(K_t + 1e-10 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            K_star = K.T  # shape (m, n)
            mean = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            var = sigma_f ** 2 - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            return mean, std

        # Helper: Expected Improvement
        def expected_improvement(mean, std, best_y, xi=0.0):
            z = (best_y - mean - xi) / std
            ei = (best_y - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Fit initial hyperparameters from data
        def fit_hyperparams(X_train, Y_train):
            """Simple heuristic: estimate length scales from data spread."""
            d = X_train.shape[1]
            # Estimate length scales as fraction of range
            ranges = X_train.max(axis=0) - X_train.min(axis=0)
            ranges = np.maximum(ranges, 1e-6)
            length_scales = 0.5 * ranges  # initial guess
            # Estimate signal variance
            sigma_f = max(np.std(Y_train), 1e-6)
            sigma_n = max(np.std(Y_train) * 0.01, 1e-6)
            return length_scales, sigma_f, sigma_n

        length_scales, sigma_f, sigma_n = fit_hyperparams(X, Y)

        # Main BO loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Predict on a candidate set
            n_cand = 200
            # Mix of random and local perturbations around best
            X_cand = rng.uniform(-5, 5, (n_cand, dim))
            # Add some points near the current best
            n_local = 50
            X_local = best_x + rng.normal(0, 0.5, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand = np.vstack([X_cand, X_local])

            # GP prediction
            try:
                mean, std = gp_predict(X_cand, X, Y, length_scales, sigma_f, sigma_n)
            except np.linalg.LinAlgError:
                # Fallback: random sampling
                x_new = rng.uniform(-5, 5, dim)
                y_new = func(x_new)
                calls_made += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, y_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                continue

            # EI acquisition
            ei = expected_improvement(mean, std, best_y, xi=0.0)
            idx = np.argmax(ei)
            x_new = X_cand[idx]

            y_new = func(x_new)
            calls_made += 1

            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Periodically refit hyperparameters
            if calls_made % 20 == 0:
                length_scales, sigma_f, sigma_n = fit_hyperparams(X, Y)

        return best_x, best_y
# EVOLVE-BLOCK-END
