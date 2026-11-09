"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        x = np.random.uniform(-5, 5, dim)
        sigma = 2.0
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        gen = 0
        while calls < budget:
            if time.time() > deadline - 5.0:
                break

            # Sample population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.randn(dim)
                L = np.linalg.cholesky(C)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()

            valid = min(lambda_, budget - calls + lambda_)
            if valid < mu:
                break
            Y = Y[:valid]
            F = F[:valid]

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Mean update
            y_mean = np.sum(weights[:valid] * Y[:min(mu, valid)], axis=0)
            x = x + y_mean

            # Update ps
            L = np.linalg.cholesky(C)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * L @ y_mean

            # Update pc
            hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * y_mean

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            diff = Y[:min(mu, valid)] - x
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) * (1 + (1 - hsig) * cc * (2 - cc)) - C) \
                + cmu * np.sum(weights[:min(mu, valid)] ** 2 * diff[:, :, None] * diff[:, None, :], axis=0)
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
