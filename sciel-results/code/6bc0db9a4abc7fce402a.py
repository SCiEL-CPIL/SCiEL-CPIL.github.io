"""Bayesian Optimization surrogate for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Gaussian Process surrogate-based optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initial random sampling
        n_init = min(15, budget)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP hyperparameters
        # We'll use a simple RBF kernel with automatic relevance determination
        # and a constant kernel
        lengthscales = np.ones(dim)
        signal_var = 1.0
        noise_var = 1e-6

        def rbf_kernel(X1, X2, lengthscales, signal_var):
            # Compute squared distances with scaling
            X1_scaled = X1 / lengthscales
            X2_scaled = X2 / lengthscales
            sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + \
                       np.sum(X2_scaled**2, axis=1)[None, :] - \
                       2 * X1_scaled @ X2_scaled.T
            sq_dists = np.maximum(sq_dists, 0)
            return signal_var * np.exp(-0.5 * sq_dists)

        def log_likelihood(X, Y, lengthscales, signal_var, noise_var):
            n = len(Y)
            K = rbf_kernel(X, X, lengthscales, signal_var) + noise_var * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            ll = -0.5 * Y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return ll

        # Acquisition: Expected Improvement (EI)
        def expected_improvement(X_new, mu, var, best_y):
            # EI for minimization
            sigma = np.sqrt(var)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (best_y - mu) / sigma
                ei = (best_y - mu) * norm.cdf(z) - sigma * norm.pdf(z)
            ei = np.where(sigma > 1e-10, ei, 0.0)
            return ei

        # Optimization loop
        calls_used = n_init
        gp_updates = 0
        max_gp_updates = 8  # Limit GP refits to avoid slow computation

        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Refit GP periodically
            if gp_updates < max_gp_updates or (calls_used - n_init) % 50 == 0:
                # Simple optimization of hyperparameters using Nelder-Mead
                # We'll do a few steps of gradient-free optimization
                def neg_ll(params):
                    ls = np.exp(params[:dim])
                    sv = np.exp(params[dim])
                    nv = np.exp(params[dim + 1])
                    return -log_likelihood(X, Y, ls, sv, nv)

                # Initial guess
                init_params = np.concatenate([np.log(lengthscales), [np.log(signal_var), np.log(noise_var)]])

                # Use scipy.optimize.minimize for a few iterations
                try:
                    from scipy.optimize import minimize
                    result = minimize(neg_ll, init_params, method='Nelder-Mead',
                                      options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
                    lengthscales = np.exp(result.x[:dim])
                    signal_var = np.exp(result.x[dim])
                    noise_var = np.exp(result.x[dim + 1])
                except Exception:
                    pass

                gp_updates += 1

            # Compute GP predictions on a candidate set
            # Use a grid + random points for acquisition maximization
            n_candidates = 200
            # Mix of grid and random
            n_grid = 100
            # Coarse grid
            grid_points = []
            for i in range(n_grid):
                grid_points.append(rng.uniform(-5, 5, dim))
            # Add random points near current best
            for i in range(n_grid):
                grid_points.append(best_x + rng.normal(0, 0.5, dim))

            X_cand = np.array(grid_points)
            X_cand = np.clip(X_cand, -5, 5)

            # GP prediction
            K = rbf_kernel(X, X, lengthscales, signal_var) + noise_var * np.eye(len(Y))
            K_s = rbf_kernel(X_cand, X, lengthscales, signal_var)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                mu = K_s @ alpha
                v = np.linalg.solve(L, K_s.T)
                var = signal_var - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: use random point
                x_new = rng.uniform(-5, 5, dim)
                f_new = func(x_new)
                calls_used += 1
                if f_new < best_y:
                    best_x = x_new
                    best_y = f_new
                X = np.vstack([X, x_new])
                Y = np.append(Y, f_new)
                continue

            ei = expected_improvement(X_cand, mu, var, best_y)
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            f_new = func(x_new)
            calls_used += 1

            if f_new < best_y:
                best_x = x_new
                best_y = f_new

            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

            # If we're close to the end, do more local exploration
            if calls_used > budget * 0.7:
                # Switch to a local search around best_x
                for _ in range(min(10, budget - calls_used)):
                    if time.time() > deadline:
                        break
                    x_try = best_x + rng.normal(0, 0.1, dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < best_y:
                        best_x = x_try
                        best_y = f_try
                    X = np.vstack([X, x_try])
                    Y = np.append(Y, f_try)

        return best_x, best_y
# EVOLVE-BLOCK-END
