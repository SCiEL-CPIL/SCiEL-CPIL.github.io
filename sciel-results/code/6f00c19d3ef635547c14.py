"""Rewritten optimizer: CMA-ES with time guard for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls = [0]

        def counted_func(x):
            calls[0] += 1
            return float(func(x))

        rng = np.random.default_rng(42)

        x0 = np.zeros(n)
        f0 = counted_func(x0)
        best_x = x0.copy()
        best_f = f0

        popsize = int(4 + 3 * np.log(n))
        popsize = max(popsize, 4)
        sigma = 2.0
        c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2))
        cs = (n + 1.3) / (n + 8)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        c_sigma = cs * (1 + ds)

        mean = x0.copy()
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        W = np.exp(np.log(0.5) / popsize)
        w = W[:popsize]
        w = w / w.sum()
        mueff = int(np.sum(w) ** 2)

        while calls[0] < budget:
            if time.time() > deadline:
                break

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Sample
            zs = rng.standard_normal((popsize, n))
            population = [mean + sigma * (sqrt_C @ z) for z in zs]
            fitnesses = []
            for x in population:
                x = np.clip(x, -5, 5)
                fx = counted_func(x)
                fitnesses.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Sort
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Update mean
            old_mean = mean.copy()
            mean = sum(w[i] * population[i] for i in range(popsize))

            # Update paths
            pc = (1 - cs) * pc + cs * np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ (mean - old_mean)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ (mean - old_mean)

            # Update sigma
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance
            delta_sig = 1 + 2 * max(0, (np.linalg.norm(ps) / chi_n) ** 2 - 1)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cs * (2 - cs) * delta_sig - 1) * C)
            for i in range(mueff):
                yi = inv_sqrt_C @ (population[i] - old_mean)
                C += cmu * w[i] * np.outer(yi, yi)

        return best_x, best_f
# EVOLVE-BLOCK-END
