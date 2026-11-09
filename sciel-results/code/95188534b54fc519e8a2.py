"""CMA-ES optimizer for Rastrigin rotated f15 with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 12, adaptive step size, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin within 300s limit
        dim = self.dim
        budget = self.budget

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # Will be updated

        # CMA-ES parameters
        mu = 6  # number of parents
        lambda_ = 12  # population size
        weights = np.log(np.arange(lambda_, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation
        fx_mean = eval_func(x_mean)

        for gen in range(1, budget + 1):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition of C for sampling
                # We maintain C via updates; use eigendecomposition for stability
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-16)
                    step = eigvecs @ (np.sqrt(eigvals) * z)
                except np.linalg.LinAlgError:
                    step = sigma * z
                y = x_mean + sigma * step
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                population.append((y, fy))

            # Check time
            if time.time() > deadline - 3.0:
                break

            # Sort by fitness
            population.sort(key=lambda p: p[1])

            # Weighted mean of best mu individuals
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i][0]

            # Standard deviation of selected points
            d = np.sqrt((x_new - x_mean).T @ np.linalg.solve(C, (x_new - x_mean)))
            if d > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / d * (x_new - x_mean) / sigma
            else:
                p_sigma = (1 - c_sigma) * p_sigma

            # Update mean
            x_mean = x_new

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update covariance matrix
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * (1 + c_c):
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            else:
                p_c = (1 - c_c) * p_c

            # Rank-one update
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * (1 + c_c) else 0
            delta_h = (1 - h_sigma) * (1 - (1 - c_c) ** (2 * gen))
            rank_one = np.outer(p_c, p_c) + delta_h

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (population[i][0] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 * h_sigma * (np.trace(C) - np.eye(dim).sum()) - c_1) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T

            # Check if we should stop (convergence)
            if sigma < 1e-12:
                break

        # Use remaining budget for local refinement if time permits
        if calls < budget and time.time() < deadline - 2.0:
            remaining = budget - calls
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(best_x + 0.1 * z, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
