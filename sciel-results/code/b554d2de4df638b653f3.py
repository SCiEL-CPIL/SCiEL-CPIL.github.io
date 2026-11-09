"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES continuous optimizer with adaptive covariance matrix and step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Initialize mean at origin
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        d = np.ones(n)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)

        while calls < budget:
            # Generate offspring
            z = np.random.randn(lam, n)
            offspring = np.array([
                np.clip(x + sigma * (B @ (d * zi)), -5, 5)
                for zi in z
            ])

            # Evaluate offspring
            fitness = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget:
                    break
                fitness[i] = func(offspring[i])
                calls += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = offspring[i].copy()

            # Sort by fitness
            order = np.argsort(fitness)
            y = np.array([offspring[order[i]] for i in range(mu)])

            # Weighted recombination (new mean)
            y_w = np.average(y, axis=0, weights=w)

            # Update path accumulators
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (d * (y_w - x)))
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (y_w - x)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = sum(wi * np.outer(yi - x, yi - x) for wi, yi in zip(w, y))
            C = (1 + c_sigma * c_sigma * (mueff - 1) * delta_h - c_1 - c_mu) * C \
                + c_1 * (rank_one + delta_h * np.eye(n)) + c_mu * rank_mu

            # Update mean
            x = y_w

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Re-orthonormalize C periodically
            if calls % 30 == 0:
                eigenvalues, B = np.linalg.eigh(C)
                d = np.sqrt(np.clip(eigenvalues, 1e-20, None))

        return best_x, best_f


# EVOLVE-BLOCK-END
