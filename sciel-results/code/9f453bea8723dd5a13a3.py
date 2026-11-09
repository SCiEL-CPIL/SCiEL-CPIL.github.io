"""CMA-ES variant with adaptive population size for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget

        # Initial population size: lambda ~ 4 + floor(3 * log(d))
        lam = max(4, 4 + int(3 * np.log(d)))
        # But we have limited budget, so keep it moderate
        lam = min(lam, 12)

        # Initialize mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, d)
        f_best = func(m)
        calls_used = 1

        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(d)
        # Covariance adaptation rates
        mu_w = lam // 2  # number of best individuals used for recombination
        weights = np.log(mu_w + 0.5) - np.log(np.arange(1, mu_w + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Path variables
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Track best
        f_opt = f_best

        # Adaptive population size: start larger, shrink as we converge
        # We'll reduce lambda over time based on progress

        while calls_used < budget:
            # Determine current population size (adaptive)
            # Start with lam, shrink as we get closer to optimum
            progress = 1.0 - (calls_used / budget)
            # Shrink population: at least 2, at most initial lam
            lam_cur = max(2, int(lam * (0.3 + 0.7 * progress)))
            lam_cur = min(lam_cur, budget - calls_used)
            if lam_cur < 1:
                break

            # Generate offspring
            offspring = []
            f_offspring = []

            for i in range(lam_cur):
                # Sample from N(m, sigma^2 * C)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)

                z = rng.standard_normal(d)
                y = m + sigma * (L @ z)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_used += 1

                offspring.append(y)
                f_offspring.append(fy)

                if fy < f_opt:
                    f_opt = fy

                if calls_used >= budget:
                    break

            if len(offspring) < 2:
                break

            # Sort by fitness
            indices = np.argsort(f_offspring)
            offspring_sorted = [offspring[i] for i in indices]
            f_sorted = [f_offspring[i] for i in indices]

            # Update mean with weighted recombination
            mu_use = min(mu_w, len(offspring_sorted))
            w_use = weights[:mu_use]
            w_use = w_use / w_use.sum()

            m_new = np.sum([w_use[i] * offspring_sorted[i] for i in range(mu_use)], axis=0)

            # Update sigma (path-based adaptation)
            diff = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff

            # Hessian potential
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_used))) < (1.4 + 1 / d) * np.sqrt(mu_eff):
                sigma *= np.exp(c_sigma / d * (np.linalg.norm(p_sigma) ** 2 / mu_eff - 1))

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(d)) +
                 c_mu * np.sum([w_use[i] * np.outer(offspring_sorted[i] - m, offspring_sorted[i] - m)
                                 for i in range(mu_use)], axis=0))

            # Ensure C is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = m_new

            # Adaptive: if we're making little progress, reduce sigma
            if f_sorted[0] < f_opt * 1.001 or (f_sorted[-1] - f_sorted[0]) < 1e-6:
                sigma *= 0.9

        return m, f_opt
# EVOLVE-BLOCK-END
