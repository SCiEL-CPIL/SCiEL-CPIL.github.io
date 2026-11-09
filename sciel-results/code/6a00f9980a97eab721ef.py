"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c1 = (2 * (mueff + 0.5)) / ((dim + 2) ** 2 + mueff + 5)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff + 5))
        cs = (mueff + 1) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial point
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0

        # CMA-ES state
        ps = np.zeros(dim)
        hs = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                pop.append(y)

            # Evaluate population
            pop_f = []
            for y in pop:
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                pop_f.append(fy)

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in order]
            pop_f_sorted = [pop_f[i] for i in order]

            # Update mean
            x_new = np.sum([weights[i] * pop_sorted[i] for i in range(mu)], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update step-size control
            y_mean = (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            norm_ps = np.linalg.norm(ps) / chi

            # Update covariance matrix
            if norm_ps < (1.4 + 2 / (dim + 1)) * (1 + 1 / damps):
                C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * np.eye(dim)) + \
                    cmu * np.sum([weights[i] * np.outer(pop_sorted[i] - x, pop_sorted[i] - x) for i in range(mu)], axis=0) / sigma ** 2

                # Eigen-decomposition (every 10 steps or so to save time, but let's do it every step for simplicity in this small dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update sigma
                sigma *= np.exp(cs / damps * (norm_ps / chi - 1))
                sigma = np.clip(sigma, 1e-10, 100)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
