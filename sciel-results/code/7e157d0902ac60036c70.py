"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES with adaptive restarts for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best across all restarts
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        lam = max(4 * dim, 10)  # population size
        # Standard CMA-ES parameters
        cc = (4 + dim / lam) / ((dim + 4) + 2 / lam)
        cs = (dim + 1.7) / (dim + dim + 2.2)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        c1 = (2 / ((dim + 1.3) ** 2 + lam)) * (1 - cc)
        cmu = min(1 - c1, 2 * (mueff - 2 / dim) / ((dim + 2) ** 2 + lam))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x_center, sigma, max_evals):
            """Run CMA-ES from x_center with initial sigma for at most max_evals evaluations."""
            nonlocal global_best_x, global_best_f

            if max_evals < lam + 1:
                # Not enough budget for one generation; do a few random steps
                x = x_center.copy()
                fx = func(x)
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
                for _ in range(max_evals - 1):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()
                    if fy < fx:
                        x, fx = y, fy
                return x, fx

            x_mean = x_center.copy()
            fx_mean = func(x_mean)
            if fx_mean < global_best_f:
                global_best_f = fx_mean
                global_best_x = x_mean.copy()

            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            evals = 1
            stagnation = 0

            while evals < max_evals:
                if time.time() > deadline:
                    break

                # Generate offspring
                B = np.linalg.cholesky(C)
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = np.clip(x_mean + sigma * B @ z, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                fitness = []
                for y in offspring:
                    fy = func(y)
                    evals += 1
                    fitness.append(fy)
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()

                fitness = np.array(fitness)
                sorted_idx = np.argsort(fitness)
                offspring_sorted = [offspring[i] for i in sorted_idx]
                fitness_sorted = fitness[sorted_idx]

                # Update mean
                x_new = sum(weights[i] * offspring_sorted[i] for i in range(mu))

                # Check convergence
                if np.all(np.abs(x_new - x_mean) < 1e-10):
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation > 5:
                    break

                # Update sigma
                p_sigma = (1 - cs) * p_sigma + cs * (B @ (x_new - x_mean) / sigma)
                sigma *= np.exp(cs / chi * (np.linalg.norm(p_sigma) / chi - 1))
                sigma = min(max(sigma, 1e-10), 10.0)

                # Update C
                p_c = (1 - cc) * p_c + cc * (x_new - x_mean) / sigma
                delta_heav = 1 + 2 * max((c1 + cmu) / (c1 - 1) - 1, 0)
                C = (1 + c1 * delta_heav - c1 - cmu) * C + cmu * np.outer(x_new - x_mean, x_new - x_mean) / sigma ** 2

                x_mean = x_new
                fx_mean = fitness_sorted[0]

            return x_mean, fx_mean

        # Restart strategy: try different starting points and sigmas
        starts = [
            (np.zeros(dim), 2.0),
            (np.full(dim, 0.5), 2.0),
            (np.full(dim, -0.5), 2.0),
            (rng.uniform(-3, 3, dim), 3.0),
            (rng.uniform(-4, 4, dim), 4.0),
        ]

        # Allocate budget: give most to first restart, rest to others
        n_restarts = min(len(starts), 5)
        # First restart gets 60% of budget, rest split evenly
        allocs = [int(budget * 0.6)] + [int(budget * 0.4 / (n_restarts - 1))] * (n_restarts - 1)
        # Adjust to sum to budget
        while sum(allocs) > budget:
            allocs[-1] -= 1
        while sum(allocs) < budget:
            allocs[0] += 1

        for i in range(n_restarts):
            if time.time() > deadline:
                break
            x0, sigma0 = starts[i]
            cma_es(x0, sigma0, allocs[i])

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
