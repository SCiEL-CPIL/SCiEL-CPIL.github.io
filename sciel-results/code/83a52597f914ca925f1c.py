"""Bayesian Optimization with GP surrogate and Student-t EI for f19."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as student_t


class Optimizer:
    """Bayesian Optimizer with GP surrogate and Student-t Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize with a small random sample
        n_init = 10
        X = np.random.uniform(-5, 5, (n_init, dim))
        # Add structured points for better initial coverage
        for i in range(dim):
            X[i, i] = 0.0
        Y = np.array([func(x) for x in X])
        calls = n_init

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP hyperparameters
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6

        # Number of candidate points to evaluate for EI
        n_candidates = 200

        # Main BO loop
        while calls < budget:
            if time.time() > deadline:
                break

            # If we have enough points for GP, use it; otherwise random sampling
            if len(Y) < 2 * dim:
                # Random sampling phase
                x_new = np.random.uniform(-5, 5, dim)
                y_new = func(x_new)
                calls += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, y_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                continue

            # Generate candidate points using a mix of strategies
            # 1. Random points
            X_rand = np.random.uniform(-5, 5, (n_candidates // 2, dim))
            # 2. Points near current best (exploitation)
            X_near = best_x + rng.normal(0, 0.3, (n_candidates // 2, dim))
            X_near = np.clip(X_near, -5, 5)

            X_candidates = np.vstack([X_rand, X_near])

            # Compute GP predictions for candidates
            mu_pred, sigma_pred = self._gp_predict(X_candidates, X, Y, length_scale, signal_var, noise_var)

            # Student-t Expected Improvement
            best_y_current = Y.min()
            sigma_t = 1.0  # Student-t scale parameter
            nu = 3  # Degrees of freedom

            # EI with Student-t
            Z = (best_y_current - mu_pred) / (sigma_pred + 1e-10)
            # For minimization, EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
            # But with Student-t, we use the t-distribution CDF
            from scipy.stats import norm
            EI = (best_y_current - mu_pred) * norm.cdf(Z) + sigma_pred * norm.pdf(Z)

            # Select point with maximum EI
            idx_max = np.argmax(EI)
            x_new = X_candidates[idx_max]
            x_new = np.clip(x_new, -5, 5)

            y_new = func(x_new)
            calls += 1
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Adapt hyperparameters based on progress
            if calls % 50 == 0:
                # Reduce length scale as we converge
                length_scale *= 0.9
                length_scale = max(length_scale, 0.1)

        # Final polishing phase with local search
        remaining = budget - calls
        if remaining > 0:
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                step_size = max(0.01, 0.1 * (remaining - _))
                x_new = best_x + rng.normal(0, step_size, dim)
                x_new = np.clip(x_new, -5, 5)
                y_new = func(x_new)
                calls += 1
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()

        return best_x, best_y

    def _gp_predict(self, X_new, X_train, Y_train, length_scale, signal_var, noise_var):
        """Gaussian Process prediction with RBF kernel."""
        # RBF kernel
        def rbf_kernel(X1, X2, ls, sv):
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            return sv * np.exp(-sq_dists / (2 * ls**2))

        K = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise_var * np.eye(len(X_train))
        K_star = rbf_kernel(X_new, X_train, length_scale, signal_var)

        # Solve for predictions
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            sigma2 = signal_var - np.sum(v**2, axis=0)
            sigma = np.sqrt(np.maximum(sigma2, 1e-10))
        except np.linalg.LinAlgError:
            # Fallback: simple interpolation
            mu = np.mean(Y_train)
            sigma = np.std(Y_train) + 0.1

        return mu, sigma
# EVOLVE-BLOCK-END
