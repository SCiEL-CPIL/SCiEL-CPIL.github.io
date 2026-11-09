"""Optimizer for BBOB f6 (Attractive sector) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with full covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # number of selected individuals

        # Initialize mean in the box
        rng = np.random.default_rng(42)
        mean = rng.uniform(-4, 4, dim)
        f_mean = func(mean)
        best_x = mean.copy()
        best_f = f_mean

        # Initial step size: roughly 20% of the box width
        sigma = 2.0

        # Covariance matrix: identity
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)

        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Step size adaptation parameters
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1)) + c_sigma

        # Covariance adaptation parameters
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_sigma_cov = (1 - c_c) * 2 * (dim + 2) / (dim + 4 + 2 * mu / dim)

        # Rank-mu update weight
        weights = np.zeros(mu)
        w_sum = np.sum(np.log((mu + 1) / 2) - np.log(np.arange(1, mu + 1)))
        for i in range(mu):
            weights[i] = np.log((mu + 1) / 2) - np.log(i + 1)
        weights /= w_sum
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Count evaluations used
        eval_count = 1  # initial point

        for gen in range(1, budget):
            # Check deadline
            if time.time() > deadline:
                break

            # Recompute eigen decomposition if needed (every 10 generations or first)
            if gen == 1 or (gen % 10 == 0):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Generate lambda offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                f_val = func(offspring[i])
                offspring_f.append(f_val)
                eval_count += 1

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Compute weighted mean of selected individuals
            y_weighted = np.zeros(dim)
            for i in range(mu):
                y_weighted += weights[i] * sorted_offspring[i]

            # Path for sigma adaptation
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y_weighted / sigma))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Path for covariance adaptation
            p_c = (1 - c_c) * p_c + \
                  np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (y_weighted / sigma))

            # Covariance matrix update
            delta_h = (1.04 + 2 / (dim + 1)) * np.linalg.norm(p_c) < (1 + c_c) * np.sqrt((dim + 1) * (2 - c_c))
            delta_h = float(delta_h)

            rank_one = np.outer(p_c, p_c)
            if delta_h:
                rank_one += (c_sigma_cov / c_sigma) * (np.outer(p_sigma, p_sigma) - np.eye(dim))

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = sorted_offspring[i] - mean
                y_i = C_inv_sqrt @ (y_i / sigma)
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_c * (delta_h * (1 - c_sigma_cov) - c_c) + c_sigma_cov * delta_h * c_sigma_cov * 0) * C \
                + c_c * (rank_one - C) + c_sigma_cov * delta_h * 0 + (1 - (1 + c_c * (delta_h * (1 - c_sigma_cov) - c_c)) - c_sigma_cov * delta_h) * 0

            # Simpler and more standard update:
            C = (1 - c_c) * C + c_c * rank_one + c_sigma_cov * delta_h * 0
            # Actually, let me use the standard formula more carefully:
            C = (1 + c_c * (delta_h * (1 - c_sigma_cov) - c_c) - c_sigma_cov * delta_h) * C \
                + c_c * rank_one + c_sigma_cov * delta_h * (np.outer(p_sigma, p_sigma) - C)

            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            # Add small regularization
            C += 1e-12 * np.eye(dim)

            # Update mean
            mean = y_weighted
            f_mean = sorted_f[0]

        return best_x, best_f
# EVOLVE-BLOCK-END
