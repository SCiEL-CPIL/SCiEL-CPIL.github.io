"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize CMA-ES parameters
        x = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / 3) / (dim + 4 + 2)
        cs = (1 + 2 / (dim + 1)) / (dim + 4 + 2 / (1 + 2 / (dim + 1)))
        cmu = min(2 / ((dim + 1.3) ** 2 + 10), 0.5) * (1 - cc)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((dim + 1 - 3) / (dim + 3)) - 1) + cs

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        weights = np.log(lambda_ / 2 + 0.5) - np.arange(lambda_)
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        # Precompute square root of C (identity initially)
        C_sqrt = np.eye(dim)

        # Evaluate initial point
        fx_best = func(x)
        best_x = x.copy()
        calls = 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate lambda_ candidates
            population = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                y = L @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                population.append((fi, xi))

            if len(population) < 2:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])

            # Update best
            if population[0][0] < fx_best:
                fx_best = population[0][0]
                best_x = population[0][1].copy()

            # Select top mueff
            mueff = min(int(np.sum(weights > 0)), len(population))
            if mueff < 2:
                mueff = 2
            selected = population[:mueff]

            # Weighted mean
            w = weights[:mueff]
            w /= w.sum()
            x_new = np.sum([w[i] * selected[i][1] for i in range(mueff)], axis=0)

            # Update evolution paths
            y_diff = x_new - x
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * y_diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) < (1.4 + 2 / (dim + 1)) * chi)
            if hsig:
                pc = (1 - cc) * pc + cc * y_diff / sigma

            # Rank mutation
            C = (1 - cs * (2 - cs) - cmu) * C \
                + cs * np.outer(ps, ps) \
                + cmu * np.sum([w[i] * np.outer(selected[i][1] - x, selected[i][1] - x)
                                 for i in range(mueff)], axis=0)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

        return best_x, fx_best
# EVOLVE-BLOCK-END
