"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim

        # Initial parameters
        x0 = rng.uniform(-3, 3, d)
        sigma = 2.0

        # CMA-ES parameters
        mu = max(1, d // 2)
        lambda_ = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        dsig = 1 + 2 * max(0, (mueff - 1) / (d + 1))

        # State variables
        mean = x0.copy()
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Cholesky factor
        try:
            Cholesky = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            Cholesky = np.eye(d)

        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1

        for gen in range(1, self.budget):
            if calls >= self.budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = mean + sigma * Cholesky @ z
                y = np.clip(y, -5, 5)
                offspring.append(y)
                calls += 1
                if calls >= self.budget:
                    break

            if len(offspring) < 2:
                break

            # Evaluate offspring
            fitnesses = []
            for y in offspring:
                fy = func(y)
                fitnesses.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(fitnesses) < mu:
                break

            # Sort by fitness
            order = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in order]
            sorted_fitnesses = [fitnesses[i] for i in order]

            # Use only the top mu offspring
            mu_actual = min(mu, len(sorted_offspring))
            parents = np.array(sorted_offspring[:mu_actual])
            w = weights[:mu_actual]
            w_sum = w.sum()
            w /= w_sum
            mueff_actual = 1.0 / np.sum(w ** 2)

            # Update mean
            new_mean = np.sum(w[:, None] * parents, axis=0)

            # Update evolution paths
            old_mean = mean.copy()
            mean = new_mean
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * (mean - old_mean) / sigma
            cs_norm = np.linalg.norm(pc) / chiN
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * (mean - old_mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) / chiN < 1.4 + 2 / (d + 1)

            # Update covariance matrix
            diff = parents - old_mean
            if hsig:
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc) - 1) * C) + cmu * np.sum(
                    w[:, None, None] * np.outer(diff, diff) / sigma ** 2,
                    axis=0
                )
            else:
                C = (1 - c1 - cmu) * C + c1 * ((1 + cc * (2 - cc) - 1) * C) + cmu * np.sum(
                    w[:, None, None] * np.outer(diff, diff) / sigma ** 2,
                    axis=0
                )

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / dsig * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Update Cholesky factor
            try:
                Cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                Cholesky = np.eye(d)

        return best_x, best_f
# EVOLVE-BLOCK-END
