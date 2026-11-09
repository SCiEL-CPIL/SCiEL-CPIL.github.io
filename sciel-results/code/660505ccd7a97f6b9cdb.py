"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents
        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        # step size adaptation (1/5 success rule on population)
        sigma = 3.0
        # covariance matrix
        C = np.eye(dim)
        # evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (sqrt_mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 - mu_eff) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def sqrt_mu_eff():
            return np.sqrt(mu_eff)

        best_x = x.copy()
        best_fx = fx

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                # Sample from N(0, C) using Cholesky
                L = np.linalg.cholesky(C)
                z = np.random.standard_normal(dim)
                Y[i] = x + sigma * (L @ z)
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Mean update
            m_new = Y_sorted[:mu] @ weights[:, np.newaxis]

            # Step size adaptation
            d_sigma = 1 + 2 * max(0, (sqrt_mu_eff() - 1) / (dim + 1)) + c_sigma
            p_sigma = (1 - c_sigma) * p_sigma + (
                np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma
            ) * (C @ (m_new - x))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Covariance adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma
            C = (1 + c_1 + c_mu - c_c) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * sum(
                w * np.outer(Y_sorted[i] - x, Y_sorted[i] - x)
                for i, w in enumerate(weights)
            ) / sigma ** 2

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-10:
                C += (1e-10 - eigvals.min()) * np.eye(dim)

            x = m_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
