"""Rewritten optimizer using Gaussian Process Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial random sampling
        n_init = min(20, budget)
        X = np.random.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_f = Y.min()

        # GP hyperparameters
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6

        def kernel(X1, X2, ls, sv):
            """RBF kernel."""
            # Compute squared distances
            X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
            X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
            sq_dists = X1_sq + X2_sq - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sv * np.exp(-sq_dists / (2 * ls ** 2))

        def predict(X_new, X_obs, Y_obs, ls, sv, nv):
            """GP prediction: mean and variance at X_new."""
            K = kernel(X_obs, X_obs, ls, sv) + nv * np.eye(len(X_obs))
            Ks = kernel(X_obs, X_new, ls, sv)
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            mu = Ks.T @ K_inv @ Y_obs
            K_ss = kernel(X_new, X_new, ls, sv)
            var = np.diag(K_ss) - np.sum(Ks.T @ K_inv * Ks, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, sigma, best_f, xi=0.01):
            """EI acquisition function."""
            sigma = np.maximum(sigma, 1e-8)
            imp = best_f - mu - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        # Main BO loop
        calls_made = n_init
        while calls_made < budget:
            # Adaptive length scale based on progress
            progress = calls_made / budget
            # Start with larger exploration, shrink later
            ls = 2.5 * (1 - progress) + 0.5 * progress

            # Generate candidate points for acquisition
            n_candidates = 200
            X_cand = np.random.uniform(-5, 5, size=(n_candidates, dim))

            # Also add points near current best
            n_near = 50
            X_near = best_x + rng.normal(0, 0.5, size=(n_near, dim))
            X_near = np.clip(X_near, -5, 5)
            X_cand = np.vstack([X_cand, X_near])

            # Predict mean and variance
            mu, var = predict(X_cand, X, Y, ls, signal_var, noise_var)
            sigma = np.sqrt(var)

            # Compute EI
            ei = expected_improvement(mu, sigma, best_f)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            f_new = func(x_new)
            calls_made += 1

            # Update observations
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
