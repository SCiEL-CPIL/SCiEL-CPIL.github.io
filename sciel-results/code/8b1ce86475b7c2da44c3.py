"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Population size as per directive
        lam = 4

        # Initial parameters
        x_mean = np.zeros(n)  # Start at origin
        sigma = 2.0           # Initial step size

        # CMA-ES parameters
        mu = lam // 2  # number of recombination points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
        eig_counter = 0
        eig_interval = int(1 / (c_1 + c_mu) / n / 10) + 1

        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1

        chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        rng = np.random.default_rng(42)

        for iteration in range(budget - calls_used):
            # Generate offspring
            x_samples = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                x_samples.append(y)

            # Evaluate offspring
            f_values = np.array([func(x) for x in x_samples])
            calls_used += lam

            # Update best
            idx_best = np.argmin(f_values)
            if f_values[idx_best] < best_f:
                best_f = f_values[idx_best]
                best_x = x_samples[idx_best].copy()

            # Sort by fitness
            idx_sorted = np.argsort(f_values)
            x_sorted = [x_samples[i] for i in idx_sorted]
            f_sorted = f_values[idx_sorted]

            # Recombination
            y_recomb = np.zeros(n)
            for i in range(mu):
                y_recomb += weights[i] * (x_sorted[i] - x_mean) / sigma

            # Update paths
            C_inv_sqrt_y_recomb = C_inv_sqrt @ y_recomb
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_y_recomb
            p_sigma_norm = np.linalg.norm(p_sigma)

            h_sigma = 1.0 if (p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) / chi_n) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_recomb

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * np.eye(n)) + c_mu * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update mean
            x_mean += sigma * y_recomb
            x_mean = np.clip(x_mean, -5, 5)

            # Periodically update eigen decomposition
            if eig_counter >= eig_interval:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    eigvals = np.ones(n)
                    eigvecs = np.eye(n)
                eig_counter = 0
            else:
                eig_counter += 1

            # Check for convergence
            if np.min(eigvals) < 1e-12 or sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
