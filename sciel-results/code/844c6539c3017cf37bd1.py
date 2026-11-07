"""Bayesian Optimization with GP surrogate and local refinement for BBOB f12."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """Derivative-free Bayesian optimization using GP surrogate + EI + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Reserve some budget for final local refinement
        n_local_refine_calls = 50
        n_bo_calls = budget - n_local_refine_calls

        # Initialize with a small set of random points
        n_init = min(10, n_bo_calls)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # GP hyperparameters
        length_scale = np.ones(dim)
        noise = 1e-6
        alpha = 1e-6

        def compute_K(X1, X2, length_scale, noise):
            """Compute GP kernel matrix (RBF)."""
            # Scale differences
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            ls_sq = length_scale ** 2
            K = np.exp(-0.5 * sq_dist / ls_sq[np.newaxis, np.newaxis, :])
            # Average over dimensions for a single length-scale-like behavior
            # Actually, let's use a single effective length scale for simplicity
            # Recompute with single length scale
            K = np.exp(-0.5 * sq_dist / (length_scale[0] ** 2))
            if X1 is X2 or np.array_equal(X1, X2):
                K = K + noise * np.eye(len(X1))
            return K

        def gp_predict(X_new, X_train, Y_train, length_scale, noise):
            """Predict mean and variance at X_new using GP."""
            K = compute_K(X_train, X_train, length_scale, noise)
            K_new = compute_K(X_train, X_new, length_scale, 0)
            K_new = K_new.T  # shape (n_new, n_train)

            # Solve for alpha
            L = np.linalg.cholesky(K + alpha * np.eye(len(X_train)))
            v = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

            # Mean
            mu = K_new @ v

            # Variance
            v_new = np.linalg.solve(L, K_new.T)
            var = 1.0 - np.sum(v_new ** 2, axis=1)
            var = np.maximum(var, 1e-12)

            return mu, var

        def expected_improvement(X_new, X_train, Y_train, best_y, length_scale, noise, xi=0.01):
            """Compute Expected Improvement."""
            mu, var = gp_predict(X_new, X_train, Y_train, length_scale, noise)
            sigma = np.sqrt(var)
            # Avoid division by zero
            sigma = np.maximum(sigma, 1e-8)
            Z = (best_y - xi - mu) / sigma
            from scipy.stats import norm
            EI = (best_y - xi - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI = np.maximum(EI, 0)
            return EI

        # Main BO loop
        for i in range(n_init, n_bo_calls):
            # Generate candidate points
            n_candidates = 2000
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))

            # Compute EI for candidates
            EI = expected_improvement(X_cand, X, Y, best_y, length_scale, noise)

            # Select best candidate
            best_cand_idx = np.argmax(EI)
            x_new = X_cand[best_cand_idx].copy()
            y_new = func(x_new)

            # Update
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Adaptively adjust length_scale based on progress
            # If we're making progress, decrease length_scale (exploit more)
            # If stuck, increase length_scale (explore more)
            if y_new < best_y + 1e-6:
                length_scale = np.maximum(length_scale * 0.9, 0.5)
            else:
                length_scale = np.minimum(length_scale * 1.05, 10.0)

        # Final local refinement using Nelder-Mead
        # Use the best point found so far
        x0 = best_x.copy()

        def objective(x):
            return func(x)

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': n_local_refine_calls // 2, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_y:
                best_x = res.x.copy()
                best_y = res.fun
        except Exception:
            pass

        # If we have budget left, do a few more local steps from best
        calls_made = n_bo_calls
        if res is not None and hasattr(res, 'nfev'):
            calls_made += res.nfev

        remaining = budget - calls_made
        if remaining > 0:
            # Do a simple local search with shrinking step
            sigma = 0.5
            for _ in range(min(remaining, 30)):
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_y:
                    best_x, best_y = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_y
# EVOLVE-BLOCK-END
