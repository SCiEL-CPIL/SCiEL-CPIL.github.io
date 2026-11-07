"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        best_x = None
        best_f = np.inf

        # Use multiple short CMA-ES runs with different starting points
        # to handle the multimodal nature of Büche-Rastrigin
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        remainder = total_budget % n_restarts

        rng_master = np.random.default_rng(42)

        for restart in range(n_restarts):
            budget_here = budget_per_restart + (1 if restart < remainder else 0)
            if budget_here <= 0:
                break

            # Start from a different point each restart
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng_master.uniform(-4, 4, dim)

            # Simple CMA-ES implementation
            x = x0.copy()
            fx = func(x)

            # Initial step size
            sigma = 1.5

            # Population size
            lambda_pop = max(4, dim + 2)

            # Covariance matrix (identity initially)
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Update rates
            mu = lambda_pop // 2
            weights = np.ones(mu)
            weights = np.sort(weights)[::-1]
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Count function calls
            calls = 1  # for x0

            for gen in range(budget_here - 1):
                if calls >= budget_here:
                    break

                # Generate population
                ys = np.zeros((lambda_pop, dim))
                fs = np.zeros(lambda_pop)

                for i in range(lambda_pop):
                    # Sample from N(x, sigma^2 * C)
                    # Simple: use sigma * standard normal (diagonal approximation for speed)
                    # For better performance, use full CMA-ES with Cholesky
                    try:
                        L = np.linalg.cholesky(C)
                        z = rng_master.standard_normal(dim)
                        y = x + sigma * L @ z
                    except np.linalg.LinAlgError:
                        y = x + sigma * rng_master.standard_normal(dim)

                    ys[i] = np.clip(y, -5, 5)
                    fs[i] = func(ys[i])
                    calls += 1

                    if calls >= budget_here:
                        break

                if i < lambda_pop - 1:
                    # Partial generation
                    lambda_pop = i + 1
                    mu = min(mu, lambda_pop)
                    weights = weights[:mu]
                    weights /= weights.sum()
                    mu_eff = 1.0 / (weights ** 2).sum()

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()
                if fs_sorted[0] < fx:
                    fx = fs_sorted[0]

                # Compute weighted mean of best mu solutions
                y_mean = np.zeros(dim)
                for i in range(min(mu, lambda_pop)):
                    y_mean += weights[i] * ys_sorted[i]

                # Update step size
                x_old = x.copy()
                x = y_mean

                # Update evolution paths
                if np.linalg.norm(x - x_old) > 1e-16:
                    diff = (x - x_old) / (sigma * np.linalg.norm(C @ (x - x_old) + 1e-16))
                    # Simplified: just use direction
                    diff = (x - x_old) / (sigma + 1e-16)
                    diff = np.clip(diff, -10, 10)

                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                    h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / np.sqrt(1 - c_sigma)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)) else 0.0

                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

                    # Update covariance
                    delta_d = (1 - h_sigma) * c_c * (2 - c_c)
                    C = (1 + c_1 + c_mu - c_1 - delta_d) * C + c_1 * (np.outer(p_c, p_c) - C)
                    for i in range(min(mu, lambda_pop)):
                        y_diff = (ys_sorted[i] - x_old) / sigma
                        C += c_mu * weights[i] * (np.outer(y_diff, y_diff) - C)

                    # Ensure positive semi-definite
                    try:
                        C = (C + C.T) / 2
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    except:
                        C = np.eye(dim)

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

                # Clip x to bounds
                x = np.clip(x, -5, 5)

        return best_x if best_x is not None else np.zeros(dim), best_f
# EVOLVE-BLOCK-END
