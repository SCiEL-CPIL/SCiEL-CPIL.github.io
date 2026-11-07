"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: random exploration then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(123)

        # Phase 1: Global exploration with 50 uniform random samples
        n_explore = min(50, total_budget // 2)
        calls_used = 0
        best_x = None
        best_f = np.inf

        # Evaluate 50 uniform random points
        for _ in range(n_explore):
            if calls_used >= total_budget:
                break
            if time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        if calls_used >= total_budget:
            return best_x, best_f

        # Phase 2: CMA-ES from the best point found
        remaining = total_budget - calls_used

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = best_x.copy()
        sigma = 1.5  # slightly larger initial step for exploration
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        C = np.eye(dim)

        iters = 0
        no_improve_count = 0
        last_best_f = best_f
        stagnation_window = 15

        while calls_used < total_budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))

            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Generate candidates
            pop = np.array([x + sigma * (C_inv_sqrt @ b) for b in B])
            pop = np.clip(pop, -5, 5)

            # Evaluate as many as we can afford
            evals_this_iter = min(lam, total_budget - calls_used)
            if evals_this_iter <= 0:
                break

            fvals = np.full(lam, np.inf)
            for i in range(evals_this_iter):
                if time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Check for improvement
            if best_f < last_best_f - 1e-10:
                no_improve_count = 0
                last_best_f = best_f
            else:
                no_improve_count += 1

            # Stagnation handling: restart CMA-ES with perturbation
            if no_improve_count >= stagnation_window and calls_used < total_budget - 10:
                # Perturb the best point and restart CMA-ES
                x = best_x + 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                sigma = max(0.5, sigma * 0.5)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                C = np.eye(dim)
                no_improve_count = 0
                last_best_f = best_f
                continue

            # Weighted mean
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update state
            y = (x_mean - x) / (sigma + 1e-15)
            p_s = (1 - c_sigma) * p_s + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sig = (np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_sigma) ** (iters + 1))) < \
                    (1 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sig * c_c * y
            rank_mu = np.outer(p_s, p_s)
            rank_one = np.outer(p_c, p_c) if h_sig else np.zeros((dim, dim))

            C = (1 + c_1 * (rank_one - C) + c_mu * rank_mu) * C
            C = (C + C.T) / 2

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_s) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            x = x_mean
            iters += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
