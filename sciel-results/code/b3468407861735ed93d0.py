"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _update_covariance(self, C, C_sqrt, C_sqrt_inv, p_c, p_sigma, 
                           y_mean, weights, mu, dim,
                           c_c, c_1, c_mu, c_sigma, d_sigma,
                           sigma, x):
        """Update CMA-ES parameters."""
        # Update p_sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu) * C_sqrt_inv @ (y_mean - x) / sigma
        sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - np.sqrt(dim) * (1 - 1 / np.sqrt(dim))))

        # Update p_c
        h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1)) < (1 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma) * mu)
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu) * C_sqrt_inv @ (y_mean - x) / sigma

        # Update C
        delta_h = (1 - h_sigma) * c_c * (2 - c_c)
        rank_one = np.outer(p_c, p_c)
        C = (1 + c_1 * (mu_eff_correction - 1) + c_mu * (mu_eff_correction - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * (y_mean - x) @ (y_mean - x).T / sigma ** 2
        # This is getting complicated, let me use a simpler approach

        return C, C_sqrt, C_sqrt_inv, p_c, p_sigma, sigma

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Initial parameters
        x = rng.uniform(-3, 3, dim)
        sigma = 1.5
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Step size control
        c_sigma = (mu + 2) / (dim + 4 + mu)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        p_sigma = np.zeros(dim)

        # Covariance matrix adaptation
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        p_c = np.zeros(dim)

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial best
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        no_improve = 0

        for gen in range(200):  # max generations
            if time.time() > deadline or calls >= budget:
                break

            # Generate population
            population = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if len(population) < mu:
                break

            # Sort by function value
            population.sort(key=lambda t: t[0])
            selected = [y for _, y in population[:mu]]

            # Weighted mean
            y_mean = np.sum([w * y for w, y in zip(weights, selected)], axis=0)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu) * C_sqrt_inv @ (y_mean - x) / sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) - 1 / (25 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update p_c
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu) * C_sqrt_inv @ (y_mean - x) / sigma

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * (mu_eff - 1) + c_mu * (mu_eff - 1) - c_1 - c_mu + c_1 * delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum([w * np.outer(y - x, y - x) for w, y in zip(weights, selected)], axis=0) / sigma ** 2

            # Update x
            x = y_mean

            # Update C_sqrt and C_sqrt_inv
            try:
                # Eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            # Check convergence
            if sigma < 1e-10:
                break

            # Early termination if no improvement
            if no_improve > 60 and sigma < 0.1:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
