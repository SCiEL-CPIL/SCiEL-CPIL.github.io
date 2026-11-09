"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin within 300s limit

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        dim = self.dim
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2  # number of selected solutions

        # Initial mean and sigma
        x_mean = np.zeros(dim)
        sigma = 2.0

        # CMA-ES weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        d_corr = 1 + 2 * max(0, (np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) / c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def update_C_sqrt():
            nonlocal C, C_sqrt, C_inv_sqrt
            # Compute matrix square root via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-18, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_C_sqrt()

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)

        budget = self.budget
        for it in range(budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            x_pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                x_new = np.clip(x_mean + sigma * C_sqrt @ z, -5, 5)
                x_pop.append(x_new)

            # Evaluate population
            f_pop = np.array([func(x) for x in x_pop])

            # Update best
            idx_min = np.argmin(f_pop)
            if f_pop[idx_min] < best_f:
                best_f = f_pop[idx_min]
                best_x = x_pop[idx_min].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            x_selected = [x_pop[i] for i in order[:mu]]
            f_selected = f_pop[order[:mu]]

            # Compute weighted mean
            x_new_mean = np.sum([weights[i] * x_selected[i] for i in range(mu)], axis=0)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((x_new_mean - x_mean) / sigma)
            p_sigma_norm = np.linalg.norm(p_sigma)

            h_sigma = float(p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1) * C
            C -= c_1 * (C - np.eye(dim))
            C += c_1 * (np.outer(p_c, p_c) + delta_h * (np.outer(p_c, p_c) - C) * 0)  # simplified
            # More standard update:
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = C_inv_sqrt @ (x_selected[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Update sigma
            sigma *= np.exp(c_sigma / d_corr * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e6)

            # Update mean
            x_mean = x_new_mean

            # Periodically update C_sqrt
            if it % 10 == 0:
                update_C_sqrt()

        return best_x, best_f
# EVOLVE-BLOCK-END
