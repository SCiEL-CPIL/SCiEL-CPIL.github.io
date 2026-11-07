"""Rewritten optimizer for BBOB f23 (Katsuura) using CMA-ES style adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive covariance matrix evolution strategy for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget

        # Initial point: use a structured initial guess
        # Katsuura has a known structure; start near origin or use random
        rng = np.random.default_rng(42)

        # Start at a reasonable point
        x = np.zeros(d)
        # Clip to box
        x = np.clip(x, -5, 5)
        fx = func(x)

        # CMA-ES parameters
        # Population size
        lambda_ = max(4, 4 + int(3 * np.log(d)))
        # Step size
        sigma = 1.0

        # Means
        m = x.copy()
        # Covariance matrix
        C = np.eye(d)
        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)
        # Eigenvalues and eigenvectors for C
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-10)
        # Inverse square root of C
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        # Square root of C
        C_sqrt = eigvecs @ np.sqrt(eigvals)

        # Update parameters
        c_sigma = (np.sqrt((lambda_ + 2) / (d + 4)) + 2) / ((d + 2) + (lambda_ + 2))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

        calls = 1
        best_x = x.copy()
        best_f = fx

        while calls < B:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lambda_):
                # Sample
                z = rng.standard_normal(d)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                if calls >= B:
                    break

            if len(offspring_f) < lambda_:
                break

            # Sort offspring
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]

            # Update mean
            m_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)

            # Update evolution paths
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (B - calls)))) < \
                   (1 + 2 / (d + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)

            # pc = (1 - c_c) * pc + hsig * c_c * (m_new - m) / sigma
            pc = (1 - c_c) * pc + (c_c if hsig else 0) * (m_new - m) / sigma

            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * sum(weights_i * (y_i - m)(y_i - m).T)
            outer_sum = np.zeros((d, d))
            for i in range(mu):
                diff = (offspring[i] - m) / sigma
                outer_sum += weights[i] * np.outer(diff, diff)

            C = (1 - c_1 - c_mu) * C + \
                c_1 * (np.outer(pc, pc) + (1 - (1 if hsig else 0)) * c_c * (2 - c_c) * C) + \
                c_mu * outer_sum

            # Symmetrize
            C = (C + C.T) / 2

            # Update eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.sqrt(eigvals)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (B - calls))) - chi_n))
            # chi_n approximation
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (25 * d ** 2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

            # Clip sigma
            sigma = np.clip(sigma, 1e-8, 10)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
