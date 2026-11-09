"""Bayesian Optimization with GP surrogate for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist


class Optimizer:
    """GP-based Bayesian Optimization for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial sampling: Latin hypercube + center
        n_init = min(20, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            X[:, d] = rng.permutation(n_init) / n_init * 10.0 - 5.0
        # Add center point
        if n_init < budget:
            X = np.vstack([X, np.zeros(dim)])
            n_init += 1

        Y = np.array([func(x) for x in X])
        n_calls = n_init

        best_x = X[np.argmin(Y)]
        best_f = Y.min()

        # GP hyperparameters
        length_scale = np.ones(dim) * 2.0
        signal_var = 1.0
        noise_var = 1e-6
        alpha = 1e-8

        def gp_predict(X_new, X_train, Y_train, ls, sv, nv):
            """Predict mean and variance of GP at X_new given training data."""
            D = cdist(X_new, X_train) ** 2
            K = sv * np.exp(-0.5 * np.sum(D * (1.0 / ls**2), axis=2)) + nv * np.eye(len(X_new))
            K_train = sv * np.exp(-0.5 * np.sum((cdist(X_train, X_train) ** 2) * (1.0 / ls**2), axis=2)) + nv * np.eye(len(X_train))
            L = np.linalg.cholesky(K_train + alpha * np.eye(len(X_train)))
            v = np.linalg.solve(L, np.linalg.solve(L.T, Y_train))
            K_star = sv * np.exp(-0.5 * np.sum((cdist(X_new, X_train) ** 2) * (1.0 / ls**2), axis=2))
            mu = K_star @ v
            v_star = np.linalg.solve(L, K_star.T)
            var = sv - np.sum(v_star**2, axis=1)
            var = np.maximum(var, 1e-12)
            return mu, var

        def ei(x, mu, var, best_f, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            if sigma < 1e-12:
                return 0.0
            z = (best_f - mu - xi) / sigma
            return (best_f - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)

        # Main BO loop
        while n_calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Adaptive hyperparameter tuning every 50 iterations
            if n_calls % 50 == 0:
                # Adjust length scales based on best region
                best_region = best_x
                spread = np.std(X, axis=0)
                length_scale = np.maximum(spread * 0.5, 0.5)
                # Adjust signal variance
                signal_var = max(np.std(Y)**2, 0.1)
                noise_var = max(1e-6, np.std(Y)**2 * 1e-4)

            # Generate candidate points
            n_candidates = 50
            X_cand = np.zeros((n_candidates, dim))
            # Mix of random, local, and exploration
            n_local = n_candidates // 3
            n_random = n_candidates // 3
            n_explore = n_candidates - n_local - n_random

            # Local around best
            X_cand[:n_local] = best_x + rng.standard_normal((n_local, dim)) * 0.5
            # Random in box
            X_cand[n_local:n_local + n_random] = rng.uniform(-5, 5, (n_random, dim))
            # Exploration: far from current points
            X_cand[n_local + n_random:] = rng.uniform(-5, 5, (n_explore, dim))
            X_cand = np.clip(X_cand, -5, 5)

            # Evaluate acquisition function
            mu, var = gp_predict(X_cand, X, Y, length_scale, signal_var, noise_var)
            ei_vals = np.array([ei(x, m, v, best_f) for x, m, v in zip(X_cand, mu, var)])

            # Select best candidate
            idx = np.argmax(ei_vals)
            x_new = X_cand[idx]
            f_new = func(x_new)
            n_calls += 1

            # Update training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new

            # Trim training set if too large (keep best 100 points)
            if len(X) > 100:
                # Keep best 80 points and most recent 20
                order = np.argsort(Y)
                keep = list(order[:80]) + list(range(len(X) - 20, len(X)))
                keep = sorted(set(keep))
                X = X[keep]
                Y = Y[keep]

        return best_x, best_f
# EVOLVE-BLOCK-END
