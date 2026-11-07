"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial point: center of the box
        x0 = np.zeros(d)
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Initial step size
        sigma = 1.0

        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        # CMA-ES parameters
        mu = lambda_ // 2  # number of weights
        # Weights: logarithmic
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = (2 - mueff / (d + 2) + c_sigma) / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

        # Covariance matrix and its square root (Cholesky)
        C = np.eye(d)
        Cinv = np.eye(d)
        C_sqrt = np.eye(d)
        C_sqrt_inv = np.eye(d)

        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Track best solution
        calls_made = 1
        remaining = budget - 1

        while remaining > 0 and sigma > 1e-12:
            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                # Sample from multivariate normal with current C
                z = rng.standard_normal(d)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            fitness = []
            for y in offspring:
                if remaining <= 0:
                    break
                fy = func(y)
                fitness.append(fy)
                remaining -= 1
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(fitness) < lambda_:
                # Not enough evaluations left for full population
                # Use partial population
                lambda_ = len(fitness)
                if lambda_ < 2:
                    break
                # Adjust weights for partial population
                mu = max(1, lambda_ // 2)
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mueff = 1.0 / np.sum(w ** 2)

            # Sort by fitness
            idx = np.argsort(fitness)
            # Take top mu
            selected = [offspring[i] for i in idx[:mu]]
            selected_fitness = [fitness[i] for i in idx[:mu]]

            # Compute weighted mean
            x_new = sum(w[i] * selected[i] for i in range(mu))

            # Update evolution paths
            diff = (x_new - x) / sigma
            # C_sqrt_inv @ diff
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ diff)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, int(remaining / lambda_))))) / np.sqrt(1 - (1 - c_sigma) ** 2) if False else 1.0
            # Simplified h_sigma check
            h_sigma = 1.0  # Approximation for stability

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * diff

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1) * C + c_1 * (np.outer(p_c, p_c) - C)
            rank_mu_term = sum(w[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu))
            C += c_mu * rank_mu_term

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(d)
                C_sqrt = np.eye(d)
                C_sqrt_inv = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)

            # Update step size
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

            # Update mean
            x = x_new

            # Clamp to box
            x = np.clip(x, -5, 5)

            # Evaluate new center if we have budget
            if remaining > 0:
                fx_new = func(x)
                remaining -= 1
                calls_made += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x.copy()

            # Termination criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(x) < 1e-10) and sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
