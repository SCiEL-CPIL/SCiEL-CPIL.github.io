"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts on a BBOB-style rotated Rastrigin (dim=5, budget=500)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 for up to budget evaluations or until deadline."""
        dim = self.dim
        n = dim
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Weights
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu = int(np.floor(lam / 2))
        wsum = weights[:mu].sum()
        mueff = int(np.floor(mu / 2))
        c_sigma = (mueff + 0.3) / (n + mueff + 3)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - c_c + mueff / (n + 2)) / ((n + 1.3) ** 2 + mueff + 2)
        c_mu = min(1 - c_1, (mueff - 2 + 2 / mueff) / ((n + 2) ** 2 + mueff - 2))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        evals = 0

        best_x = x.copy()
        best_f = func(x)
        evals += 1

        while evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            fitnesses = np.array([func(y) for y in offspring])
            evals += lam

            # Update best
            idx_best = np.argmin(fitnesses)
            if fitnesses[idx_best] < best_f:
                best_f = fitnesses[idx_best]
                best_x = offspring[idx_best].copy()

            # Sort
            order = np.argsort(fitnesses)
            y_sorted = [offspring[i] for i in order]
            f_sorted = fitnesses[order]

            # Update mean
            y_mu = np.average(y_sorted[:mu], axis=0, weights=weights[:mu])
            d = y_mu - x
            x = np.clip(x + d, -5, 5)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * d))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals / (n + 1))) < (1.4 + 1 / n) * np.sqrt(mueff / (n + 2)):
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))

            # Update covariance
            delta_mu = c_1 * (1 + (1 - c_c) * np.dot(p_c, d) / np.linalg.norm(d) ** 2 if np.linalg.norm(d) > 1e-12 else 0)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * d
            outer = np.outer(d, d)
            outer += (c_1 / c_c) * (1 - c_c) * np.outer(p_c, p_c)
            # Simplified: use rank-1 and rank-mu update
            H = np.zeros((n, n))
            for i in range(mu):
                H += weights[i] * np.outer(y_sorted[i] - x, y_sorted[i] - x)
            H /= wsum
            C_new = (1 + c_1 * (2 - c_1) - c_mu) * np.eye(n) + c_1 * (np.outer(p_c, p_c) - np.eye(n)) + c_mu * H
            # Ensure positive semi-definite
            C_new = (C_new + C_new.T) / 2
            # Eigen decomposition for B and D
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs
            D = np.sqrt(eigvals)
            sigma = np.clip(sigma, 1e-12, 100)

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Multi-start strategy with CMA-ES
        # First try a few random starts with small budgets to find a good region
        n_starts = 5
        per_budget = max(1, budget // n_starts)

        # Initial diverse starts
        starts = []
        starts.append(np.zeros(dim))  # Center
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-4, 4, dim))

        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            remaining_budget = budget - (i * per_budget)
            if remaining_budget < 10:
                break
            x0 = np.clip(x0, -5, 5)
            fx0 = func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Run CMA-ES from this start
            sigma0 = 1.0
            bx, bf = self._cma_es(func, x0, sigma0, per_budget, rng, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
