"""CMA-ES optimizer for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based search and covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # CMA-ES parameters
        lam = max(2 * dim + 2, 8)  # population size
        mu = lam // 2              # number of recombination weights

        # Weights for mean update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial values
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        # Cholesky factor of C
        C_sqrt = np.eye(dim)
        # Step size control
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial population
        rng = np.random.default_rng(42)
        y = rng.standard_normal((lam, dim))
        x_pop = np.clip(x_mean + sigma * y @ C_sqrt.T, -5, 5)
        f_pop = np.array([func(x) for x in x_pop])

        # Track best
        best_x = x_pop[np.argmin(f_pop)].copy()
        best_f = f_pop.min()

        calls_made = lam

        while calls_made < budget and time.time() < deadline:
            # Sort by fitness
            order = np.argsort(f_pop)
            y = y[order]
            f_pop = f_pop[order]
            y_w = y[:mu]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y_w, axis=0)

            # Update path for covariance
            C_inv_sqrt = np.linalg.inv(C_sqrt)
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            h_sig = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (calls_made + 1)) < (1 + 2 / (dim + 1)) * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_old) / sigma

            # Update covariance
            delta_sig = h_sig * (np.linalg.norm(p_sigma) / chi_n - 1)
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
                 + c_mu * np.sum(weights[:, None, None] * np.outer(y_w, y_w), axis=0))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += (1e-12 - eigvals.min()) * np.eye(dim)

            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)

            # Generate new population
            y = rng.standard_normal((lam, dim))
            x_pop = np.clip(x_mean + sigma * y @ C_sqrt.T, -5, 5)
            f_pop = np.array([func(x) for x in x_pop])
            calls_made += lam

            # Update best
            idx = np.argmin(f_pop)
            if f_pop[idx] < best_f:
                best_f = f_pop[idx]
                best_x = x_pop[idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
