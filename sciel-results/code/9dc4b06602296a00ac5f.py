"""Rewritten optimizer using CMA-ES style adaptive strategy (from scratch)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f17 (Schaffers F7) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial population size and sigma
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        # Start with a reasonable sigma
        sigma = 2.0

        # Initial center: random point in the box
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (1 - c_c + c_sigma)
        c_mu = (2 / ((n + 2) ** 2 + mu_eff)) * (min(1 - c_1, 2 * (weights[0] - 1 / mu_eff)))
        c_mu = min(c_mu, 1 - c_1)

        # Covariance matrix
        C = np.eye(n)
        # Step sizes for evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Track best
        call_count = 1

        def clip(x):
            return np.clip(x, -5, 5)

        while call_count < budget:
            # Compute step size matrix
            # sqrt(C) via eigendecomposition
            try:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-20)
                sqrt_C = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
            except np.linalg.LinAlgError:
                sqrt_C = np.eye(n)

            # Generate lambda_pop offspring
            offspring = []
            offspring_f = []
            for i in range(lambda_pop):
                y = x + sigma * (sqrt_C @ rng.standard_normal(n))
                y = clip(y)
                fy = func(y)
                call_count += 1
                offspring.append(y)
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                if call_count >= budget:
                    break

            if call_count >= budget:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]

            # Weighted mean
            m_new = np.sum(weights[:, None] * offspring[:mu], axis=0)

            # Update evolution paths
            # p_sigma
            inv_sqrt_C = np.linalg.pinv(sqrt_C)
            delta = (m_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ delta)

            # p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count // lambda_pop)))) < (1.4 + 2 / (n + 1)) * np.sqrt((1 - c_sigma) / (1 - (1 - c_sigma) ** (2 * (call_count // lambda_pop))))

            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * (h_sigma * (1 + 1 / c_1) - 1) - c_1 - c_mu) * C \
                + c_1 * (outer - (1 - h_sigma) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(offspring[:mu] - x, offspring[:mu] - x) / sigma ** 2, axis=0)

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Regularize
            C += 1e-12 * np.eye(n)

            # Update center
            x = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
