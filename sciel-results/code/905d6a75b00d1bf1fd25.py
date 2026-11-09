"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Restart (1+1)-ES for exploration
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        stagnation_counter = 0
        stagnation_threshold = 20

        # Run (1+1)-ES for about 60% of budget
        phase1_budget = int(budget * 0.6)
        calls_used = 0

        while calls < budget and calls_used < phase1_budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            calls_used += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                stagnation_counter = 0
            else:
                sigma *= 0.85
                stagnation_counter += 1

            if stagnation_counter >= stagnation_threshold:
                # Restart from a new random point
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                calls_used += 1
                sigma = 2.0
                stagnation_counter = 0

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: CMA-ES refinement around the best point found
        remaining = budget - calls
        if remaining > 10:
            cma_best_x, cma_best_f, cma_calls = self._run_cmaes(func, best_x, best_f, remaining, rng, deadline)
            calls += cma_calls
            if cma_best_f < best_f:
                best_f = cma_best_f
                best_x = cma_best_x.copy()

        return best_x, best_f

    def _run_cmaes(self, func, x0, f0, max_calls, rng, deadline):
        """Run CMA-ES from x0 with budget max_calls. Returns (best_x, best_f, calls_made)."""
        dim = self.dim
        lam = 6
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()

        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2 * min(1, 0.4 / (dim + 1))) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))

        x_mean = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        local_calls = 0
        local_best_x = x0.copy()
        local_best_f = f0

        while local_calls < max_calls:
            if time.time() > deadline:
                break

            offspring = []
            offspring_f = []
            for i in range(lam):
                if local_calls >= max_calls:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                offspring_f.append(fy)
                local_calls += 1

                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()

            if len(offspring_f) < 2:
                break

            idx = np.argsort(offspring_f)
            sorted_off = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]

            old_mean = x_mean.copy()

            x_mean = np.zeros(dim)
            for i in range(mu):
                x_mean += weights[i] * sorted_off[i]

            if sigma > 1e-10:
                v_mean = (x_mean - old_mean) / sigma
            else:
                v_mean = np.zeros(dim)

            h_sigma = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * v_mean
            sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-8, 10)

            h_c = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)
            p_c = (1 - c1 - cmu) * p_c + h_c * np.sqrt((c1 + cmu) * (2 - c1 - cmu) * mueff) * v_mean

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                if sigma > 1e-10:
                    vi = (sorted_off[i] - old_mean) / sigma
                else:
                    vi = np.zeros(dim)
                rank_mu += weights[i] * np.outer(vi, vi)

            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu
            C = (C + C.T) / 2

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 1e-3)

            if np.linalg.norm(x_mean - old_mean) < 1e-10 * sigma:
                break

        return local_best_x, local_best_f, local_calls
# EVOLVE-BLOCK-END
