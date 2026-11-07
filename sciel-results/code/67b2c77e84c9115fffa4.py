"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance and stagnation-aware restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline, best_x, best_f):
        """Run CMA-ES for up to `budget` evaluations, respecting deadline."""
        dim = self.dim
        n = dim
        # CMA-ES parameters
        mu = max(1, n // 2)  # number of selected parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / n) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        calls = 0

        # Initial population
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                return best_x, best_f

            # Generate population
            pop = np.empty((mu * 2, n))
            for i in range(mu * 2):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                # Whitening via Cholesky of C
                L = np.linalg.cholesky(C)
                y = L @ z
                pop[i] = x + sigma * y
                pop[i] = np.clip(pop[i], -5, 5)
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Evaluate
            fvals = np.full(mu * 2, np.inf)
            for i in range(min(calls, mu * 2)):
                if time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            idx = np.argsort(fvals)[:mu]
            if len(idx) < mu:
                break

            # Weighted mean
            m = np.average(pop[idx], axis=0, weights=w)

            # Update step size
            y_sigma = (m - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 50.0)

            # Update evolution path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = ((1 - c_1 - c_mu + delta_h) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (2 - c_c) * C) +
                 c_mu * sum(w[i] * np.outer(pop[idx[i]] - x, pop[idx[i]] - x) for i in range(mu)))

            # Update mean
            x = m
            x = np.clip(x, -5, 5)

            # Symmetrize C and ensure positive definiteness
            C = (C + C.T) / 2
            # Eigenvalue clipping for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(n)

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Multi-start with diverse initial points
        n_starts = 8
        remaining = self.budget
        per_start = max(30, remaining // n_starts)

        for s in range(n_starts):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Generate diverse starting point
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif s == 2:
                x0 = rng.uniform(-4, 4, dim)
            elif s == 3:
                x0 = rng.choice([-4, -2, 0, 2, 4], dim)
            elif s == 4:
                x0 = rng.uniform(-5, 5, dim)
            elif s == 5:
                # Try around origin with small perturbation
                x0 = rng.uniform(-1, 1, dim)
            elif s == 6:
                x0 = rng.uniform(-3, 3, dim)
                x0 += rng.standard_normal(dim) * 0.5
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)

            x0 = np.clip(x0, -5, 5)

            # Evaluate initial point
            fx = func(x0)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

            # Run CMA-ES from this point
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break

            sigma0 = 1.5
            budget_here = min(per_start, self.budget - calls_used)

            best_x, best_f = self._cmaes(
                func, x0, sigma0, budget_here, rng, deadline, best_x, best_f
            )
            calls_used += budget_here

            # If we found a good point, do a local refinement
            if best_f < 50 and calls_used < self.budget - 10:
                if time.time() > deadline:
                    break
                # Local search with smaller sigma
                budget_local = min(30, self.budget - calls_used)
                best_x, best_f = self._cmaes(
                    func, best_x, 0.3, budget_local, rng, deadline, best_x, best_f
                )
                calls_used += budget_local

        # Final check: if we have time, do one more refinement from the best point
        if calls_used < self.budget - 5 and time.time() < deadline:
            budget_final = min(20, self.budget - calls_used)
            if best_x is not None:
                best_x, best_f = self._cmaes(
                    func, best_x, 0.2, budget_final, rng, deadline, best_x, best_f
                )

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
