"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial step size
        sigma = 2.0

        # Initial mean: start at origin (common for BBOB-style functions)
        mean = np.zeros(n)

        # Initial covariance matrix
        C = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Cholesky factor of C
        C_chol = np.linalg.cholesky(C)

        # Count function evaluations used
        calls = 0

        def safe_func(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # Initial evaluation
        f_mean = safe_func(mean)
        best_x = mean.copy()
        best_f = f_mean

        # Main loop
        while calls < budget:
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                z = np.random.randn(n)
                x_new = mean + sigma * (C_chol @ z)
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)

            # Evaluate offspring
            offspring_f = [safe_func(x) for x in offspring]

            # Update best
            for x, f in zip(offspring, offspring_f):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Sort offspring by fitness
            sorted_idx = np.argsort(offspring_f)
            offspring_sorted = [offspring[i] for i in sorted_idx]
            offspring_f_sorted = [offspring_f[i] for i in sorted_idx]

            # Recombination: weighted mean of best mu offspring
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * offspring_sorted[i]

            # Update evolution paths
            # Normalized step
            step = (new_mean - mean) / (sigma * np.linalg.norm(C_chol)) if np.linalg.norm(C_chol) > 1e-10 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) <
                       (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma) ** (budget - calls)) * c_sigma))
            h_sigma = float(h_sigma)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step

            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                z_i = (offspring_sorted[i] - mean) / sigma if sigma > 1e-10 else np.zeros(n)
                rank_mu += weights[i] * np.outer(z_i, z_i)

            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * (rank_one - (1 - h_sigma * c_c * (2 - c_c)) * C) + c_mu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_chol = np.eye(n)

            # Update step size
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget - calls)) - np.sqrt(mu_eff)))

            # Update mean
            mean = new_mean

            # Early stopping if progress is very small
            if budget - calls < 10 and sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
