"""CMA-ES optimizer with deadline guard for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # standard CMA-ES lambda
        # Weights for selection
        mu = lam // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Chi distribution
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: center of the box
        x_mean = np.zeros(dim)
        sigma = 2.0  # initial step size
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = np.inf

        # Helper to compute matrix square root and inverse square root
        def update_C_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_inv, C_inv_sqrt

        C_sqrt, C_inv, C_inv_sqrt = update_C_sqrt(C)

        # Call counter
        call_count = 0

        def make_call(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_x = x.copy()
                best_f = f
            return x, f

        # Initial evaluation
        x0, f0 = make_call(x_mean)

        # Main loop
        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                z = np.random.randn(dim)
                x_new = x_mean + sigma * C_sqrt @ z
                x_new, f_new = make_call(x_new)
                population.append((f_new, x_new))

            if time.time() > deadline:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])

            # Update mean
            w = weights
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * population[i][1]

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 50.0)

            # Update C
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sigma = 1
            else:
                h_sigma = 0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma

            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = C_inv_sqrt @ (population[i][1] - x_mean) / sigma
                rank_mu += w[i] * np.outer(z_i, z_i)

            C = (1 + c_1 * h_sigma * (dim + 1) - c_1 - c_mu) * C + c_1 * (rank_1 - (1 - h_sigma) * (1 - c_c) * C) + c_mu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C, _ = np.linalg.eigh(C)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            C_sqrt, C_inv, C_inv_sqrt = update_C_sqrt(C)

            x_mean = x_new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
