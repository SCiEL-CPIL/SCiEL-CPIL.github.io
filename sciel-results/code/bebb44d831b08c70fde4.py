"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step-size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        x0 = np.random.default_rng(42).uniform(-3, 3, n)
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial population
        rng = np.random.default_rng(123)
        lambda_ = 2 * n
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Precompute some constants
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lambda_, n))
            offspring_f = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x0 + sigma * (C_sqrt @ z)
                offspring[i] = np.clip(y, -5, 5)
                offspring_f[i] = func(offspring[i])
                calls += 1

            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            offspring_sorted = offspring[idx]
            offspring_f_sorted = offspring_f[idx]

            # Update best
            if offspring_f_sorted[0] < best_f:
                best_f = offspring_f_sorted[0]
                best_x = offspring_sorted[0].copy()

            # Update mean
            old_mean = x0.copy()
            x0 = sum(weights[i] * offspring_sorted[i] for i in range(mu))

            # Update covariance matrix
            diff = (x0 - old_mean) / sigma
            p_c = (1 - c_c) * p_c + c_c * diff
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ diff)

            h_sigma = 0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1 + c_sigma) * chi_n:
                h_sigma = 1

            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(offspring_sorted[i] - old_mean, offspring_sorted[i] - old_mean) for i in range(mu))
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c) - c_1) - c_1 - c_mu) * C \
                + c_1 * rank_one + c_mu * rank_mu

            # Re-estimate C_sqrt via Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize and retry
                C += 1e-10 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Check convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
