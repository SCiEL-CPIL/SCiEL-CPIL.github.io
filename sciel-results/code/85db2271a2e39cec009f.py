"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance and population-based search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(n)) for n=5 -> 4 + floor(3*1.609) = 4+4 = 8
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = int(np.sum(weights[:mu]) ** 2 / np.sum(weights ** 2))

        # Initial sigma
        sigma = 2.0
        # Initial mean: center of the box
        mean = np.zeros(dim)

        # Covariance matrix C = I
        C = np.eye(dim)
        # Precompute C^(1/2) and C^(-1/2)
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
        C_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Step sizes
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_mu = (2 + weights[mu - 1]) / (dim + 2 + weights[mu - 1]) if mu > 1 else (2 / (dim + 2))

        # Count evaluations
        eval_count = 0

        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        best_x = x0.copy()
        best_f = func(x0)
        eval_count += 1

        # Main loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            offspring = []
            fitnesses = []
            for i in range(lam):
                # Sample
                z = np.random.randn(dim)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                fitnesses.append(fy)
                eval_count += 1

                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                # Early termination if time is tight
                if eval_count >= budget or time.time() > deadline - 5:
                    break

            if eval_count >= budget or time.time() > deadline - 5:
                break

            # Sort offspring by fitness
            order = np.argsort(fitnesses)
            offspring_sorted = [offspring[i] for i in order]
            fitnesses_sorted = [fitnesses[i] for i in order]

            # Update mean
            new_mean = sum(weights[i] * offspring_sorted[i] for i in range(mu))

            # Compute weighted mean of selected solutions
            # Update evolution paths
            y_vec = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_invsqrt @ y_vec)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - eval_count) / lam))
                       < (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_vec

            # Update covariance
            delta = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_c
            C = ((1 + c_c * (2 - c_c) - c_mu) * C
                 + c_c * np.outer(p_c, p_c)
                 + c_mu * sum(weights[i] * np.outer(offspring_sorted[i] - mean, offspring_sorted[i] - mean)
                              for i in range(mu)))

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update square root
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
