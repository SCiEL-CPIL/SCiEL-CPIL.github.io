"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s kill

        n = self.dim
        # CMA-ES parameters
        popsize = max(4, int(4 + 3 * np.log(n)))
        lambda_ = popsize
        mu = lambda_ // 2

        # Initial point: center of search space
        x_best = np.zeros(n)
        f_best = func(x_best)

        # Initial covariance: identity, sigma: large to explore
        sigma = 2.5
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Learning rates
        c_sigma = (mu + 2) / (n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu) / ((n + 2) ** 2 + mu_eff))
        mu_eff = 1 / np.sum(1.0 / np.arange(1, mu + 1))
        # Recompute c_mu with correct mu_eff
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu) / ((n + 2) ** 2 + mu_eff))

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        calls_made = 1

        for gen in range(1, self.budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            f_offspring = []
            for i in range(lambda_):
                # Cholesky-based sampling
                z = np.random.standard_normal(n)
                y = C_sqrt @ z
                x_new = np.clip(x_best + sigma * y, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                offspring.append(x_new)
                f_offspring.append(f_new)

                if calls_made >= self.budget:
                    break

            if len(offspring) == 0:
                break

            # Sort by fitness
            idx = np.argsort(f_offspring)
            sorted_f = [f_offspring[i] for i in idx]
            sorted_x = [offspring[i] for i in idx]

            # Update best
            if sorted_f[0] < f_best:
                x_best = sorted_x[0]
                f_best = sorted_f[0]

            # Use top mu for update
            x_mean = np.zeros(n)
            for i in range(min(mu, len(sorted_x))):
                x_mean += weights[i] * sorted_x[i]

            # Update sigma (CSA)
            y_sigma = (x_mean - x_best) / sigma if sigma > 0 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance (CMSA)
            y_c = (x_mean - x_best) / sigma if sigma > 0 else np.zeros(n)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            delta_h = (1 + c_c) * np.linalg.norm(p_c) / np.sqrt(mu_eff) - (1 - c_c)
            h_sig = 1.0 if delta_h < 1.4 + 2 / (n + 1) else 0.0

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C) \
                + c_mu * sum(weights[i] * np.outer(sorted_x[i] - x_best, sorted_x[i] - x_best) / sigma**2 for i in range(min(mu, len(sorted_x))))

            # Re-regularize C periodically
            if gen % 20 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_sqrt via SVD (simple approach)
            U, S, Vt = np.linalg.svd(C)
            C_sqrt = (U * np.sqrt(S)) @ Vt
            C_inv = np.linalg.inv(C + 1e-10 * np.eye(n))

            # Move mean
            x_best = x_mean

            if calls_made >= self.budget:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
