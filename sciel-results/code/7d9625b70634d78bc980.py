"""Optimizer for BBOB f19 (Griewank-Rosenbrock) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES for a given budget, respecting deadline."""
        n = self.dim
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        best_x = x.copy()
        best_f = func(x)
        evals = 1

        while evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = np.array([rng.multivariate_normal(np.zeros(n), C, size=1).flatten()
                           for _ in range(lambda_)])
            ys = sigma * ys
            xs = x + ys

            # Evaluate
            fs = np.array([func(xi) for xi in xs])
            evals += lambda_

            # Update best
            if fs.min() < best_f:
                best_f = fs.min()
                best_x = xs[np.argmin(fs)].copy()

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            ys_sorted = ys_sorted[:mu]

            # Weighted mean
            m = np.sum(weights[:, None] * ys_sorted, axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * np.sqrt(np.diag(C).sum() / n)) * m
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals // lambda_ + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * m / sigma

            # Rank-one and rank-mu updates
            if hsig:
                C = (1 - c_1 - c_mu) * C + (1 + c_1 - c_mu) * np.outer(pc, pc)
            else:
                C = (1 - c_1 - c_mu) * C + c_1 * np.eye(n)
            C += c_mu * np.sum(weights[:, None] * np.outer(ys_sorted, ys_sorted), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Step size adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            # Update mean
            x = x + m

            # Clip to bounds
            x = np.clip(x, -5, 5)

        return best_x, best_f

    def __call__(self, func):
        deadline = time.time() + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        n = self.dim

        best_x = None
        best_f = np.inf

        # Try multiple restarts with different initializations
        n_restarts = 5
        budget_per_restart = max(10, self.budget // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if r == 0:
                x0 = rng.uniform(-5, 5, n)
            elif r == 1:
                x0 = np.zeros(n)
            elif r == 2:
                x0 = rng.uniform(-3, 3, n)
            elif r == 3:
                x0 = rng.uniform(-1, 1, n)
            else:
                x0 = rng.uniform(-5, 5, n)

            sigma0 = 2.0 if r == 0 else 1.0

            # Remaining budget
            remaining = self.budget - (r * budget_per_restart)
            if remaining <= 0:
                break

            x_opt, f_opt = self._cma_es(func, x0, sigma0, remaining, deadline, rng)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        # If no valid result, do a simple random search
        if best_x is None:
            x = rng.uniform(-5, 5, n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
