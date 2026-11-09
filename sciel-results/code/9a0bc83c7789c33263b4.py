"""Deterministic coarse grid scan followed by localized CMA-ES for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse grid scan to locate basin, then CMA-ES for local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Phase 1: Coarse grid scan (~243 calls) to locate the global basin
        # 3^5 = 243 points on a 3x3x3x3x3 grid over [-5, 5]^5
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([-5 + 10 * i / 2,
                                          -5 + 10 * j / 2,
                                          -5 + 10 * k / 2,
                                          -5 + 10 * l / 2,
                                          -5 + 10 * m / 2])
                            grid_points.append(x)

        best_x = None
        best_f = float('inf')
        calls = 0

        for x in grid_points:
            if time.time() > deadline:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: CMA-ES starting from the grid's best point
        remaining = self.budget - calls
        if remaining > 5:
            x, f = self._cma_es(func, best_x, best_f, remaining, deadline)
            return x, f
        else:
            return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, deadline):
        """CMA-ES implementation for local exploitation."""
        n = self.dim
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        sigma = 1.0  # initial step size
        mu = max(1, n // 2)  # number of parents
        lam = 2 * mu  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Covariance matrix and evolution path
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = x0.copy()
        fx = f0
        calls = 0

        for gen in range(budget):
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, n))
            # Whitening transform
            D = np.linalg.cholesky(C)
            y = B @ D.T
            offsprings = x + sigma * y

            # Evaluate
            offspring_f = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                xi = np.clip(offsprings[i], -5, 5)
                offspring_f[i] = func(xi)
                calls += 1

            # Sort and select
            idx = np.argsort(offspring_f)
            y_sorted = y[idx[:mu]]
            f_sorted = offspring_f[idx[:mu]]
            x_new = (weights * offsprings[idx[:mu]]).sum(axis=0)
            f_new = offspring_f[idx[0]]

            # Update mean
            if f_new < fx:
                x, fx = np.clip(x_new, -5, 5), f_new

            # Update evolution paths
            y_mean = (weights * y_sorted).sum(axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D @ y_mean) / sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (D @ y_mean) / sigma

            # Update covariance matrix
            delta_h = (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1) / (n + 1)) - 1)
            C = ((1 + c_1 * delta_h - c_1 - c_mu) * C +
                 c_1 * (np.outer(pc, pc) - C) +
                 c_mu * np.outer(weights * (y_sorted / sigma), weights * (y_sorted / sigma)).sum(axis=0) / sigma ** 2)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Safety: clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
