"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a small random sample
        n_init = 10
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])

        # GP hyperparameters
        lengthscales = np.ones(dim) * 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def compute_kernel(X1, X2, ls):
            """Compute Matern-5/2 kernel between X1 and X2."""
            if np.isscalar(ls):
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist) / ls
            else:
                scaled_X1 = X1 / ls
                scaled_X2 = X2 / ls
                diff = scaled_X1[:, np.newaxis, :] - scaled_X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist)

            return signal_var * (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)

        X_list = X.copy()
        Y_list = Y.copy()

        # Pre-compute a coarse grid for acquisition function maximization
        grid_vals = np.linspace(-5, 5, 5)
        grid = np.array(np.meshgrid(*[grid_vals] * dim)).T.reshape(-1, dim)

        for i in range(budget - n_init):
            n = len(Y_list)
            K = compute_kernel(X_list, X_list, lengthscales)
            K += noise_var * np.eye(n)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)

            v = np.linalg.solve(L, Y_list)
            alpha = np.linalg.solve(L.T, v)

            def neg_ei(x):
                x = x.reshape(1, -1)
                k_x = compute_kernel(x, X_list, lengthscales)
                mu = (k_x @ alpha)[0]
                v_x = np.linalg.solve(L, k_x.T)
                var = signal_var - np.sum(v_x * v_x)
                var = max(var, 1e-10)
                sigma = np.sqrt(var)
                y_best = np.min(Y_list)
                xi = 0.01
                z = (mu - y_best - xi) / sigma
                ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Evaluate EI on grid
            k_grid = compute_kernel(grid, X_list, lengthscales)
            mu_grid = k_grid @ alpha
            v_grid = np.linalg.solve(L, k_grid.T)
            var_grid = signal_var - np.sum(v_grid * v_grid, axis=0)
            var_grid = np.maximum(var_grid, 1e-10)
            sigma_grid = np.sqrt(var_grid)

            y_best = np.min(Y_list)
            xi = 0.01
            z_grid = (mu_grid - y_best - xi) / sigma_grid
            ei_grid = (mu_grid - y_best - xi) * norm.cdf(z_grid) + sigma_grid * norm.pdf(z_grid)

            # Select top grid points for local optimization
            top_idx = np.argsort(ei_grid)[-10:]
            best_neg_ei = np.inf
            best_x = grid[top_idx[0]]

            for idx in top_idx:
                x0 = grid[idx]
                try:
                    res = minimize(neg_ei, x0, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * dim,
                                   options={'maxiter': 50, 'ftol': 1e-6})
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_x = res.x
                except Exception:
                    pass

            # Also try a few random restarts
            for _ in range(5):
                x0 = rng.uniform(-5, 5, dim)
                try:
                    res = minimize(neg_ei, x0, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * dim,
                                   options={'maxiter': 50, 'ftol': 1e-6})
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_x = res.x
                except Exception:
                    pass

            x_new = best_x
            y_new = func(x_new)

            X_list = np.vstack([X_list, x_new])
            Y_list = np.append(Y_list, y_new)

            # Occasionally adapt length scale
            if i % 50 == 0 and i > 0:
                if len(Y_list) > 20:
                    recent_y = Y_list[-20:]
                    if np.std(recent_y) < 0.5:
                        lengthscales = np.maximum(lengthscales * 0.8, 0.1)

        return X_list[-1], Y_list[-1]
# EVOLVE-BLOCK-END
