"""Rastrigin optimizer: Custom CMA-ES with adaptive restarts + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Custom CMA-ES with adaptive restarts for multimodal Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Broad initial sampling ---
        n_init = 20
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # --- Phase 2: Custom CMA-ES with adaptive restarts ---
        remaining = budget - calls_made
        if remaining > 30 and time.time() < deadline:
            self._cma_es_search(counted_func, best_x.copy(), 2.0, remaining, dim, rng, deadline)

        # --- Phase 3: Local polish ---
        if calls_made < budget - 10 and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func, best_x, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Final perturbations + polish ---
        if calls_made < budget - 5 and time.time() < deadline - 3:
            for _ in range(3):
                if calls_made >= budget - 3 or time.time() > deadline - 3:
                    break
                x_perturb = best_x + rng.normal(0, 0.05, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                try:
                    res = minimize(
                        counted_func, x_perturb, method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f

    def _cma_es_search(self, func, x0, sigma0, max_calls, dim, rng, deadline):
        """Clean CMA-ES with adaptive restarts."""
        nonlocal calls_made, best_x, best_f

        lam = 30
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = x0.copy().astype(np.float64)
        sigma = sigma0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        local_best_f = best_f
        local_best_x = best_x.copy()
        stagnation = 0
        max_stagnation = lam * 5
        total_calls = 0
        gen = 0

        while total_calls < max_calls:
            if time.time() > deadline:
                break

            if stagnation >= max_stagnation:
                x_mean = rng.uniform(-4, 4, dim)
                sigma = max(1.0, sigma * 0.5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation = 0
                gen = 0
                fx = func(x_mean)
                total_calls += 1
                if np.isfinite(fx) and fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x_mean.copy()
                continue

            # Generate offspring
            B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            D = np.sqrt(np.diag(C))

            offspring = []
            fitnesses = []
            for i in range(lam):
                if total_calls >= max_calls:
                    break
                z = rng.standard_normal(dim)
                x_new = x_mean + sigma * (B @ z)
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                total_calls += 1
                offspring.append(x_new)
                fitnesses.append(fx)
                if np.isfinite(fx) and fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x_new.copy()

            if len(offspring) == 0:
                break

            fitnesses_arr = np.array(fitnesses)
            offspring_arr = np.array(offspring)
            idx = np.argsort(fitnesses_arr)
            offspring_arr = offspring_arr[idx]
            fitnesses_arr = fitnesses_arr[idx]

            if fitnesses_arr[0] < local_best_f - 1e-10:
                stagnation = 0
            else:
                stagnation += 1

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum(w[:, None] * offspring_arr[:mu], axis=0)

            # Update p_sigma
            y_mean = (x_mean - old_mean) / (sigma * np.sqrt(np.diag(C)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update p_c
            y_c = (x_mean - old_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c

            # Update covariance
            delta = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) * c_c if False else 0
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(dim)) +
                 c_mu * np.sum(w[:, None, None] * np.outer(offspring_arr[:mu] - old_mean,
                    (offspring_arr[:mu] - old_mean).T), axis=0) / sigma ** 2)

            gen += 1

        if local_best_f < best_f:
            best_f = local_best_f
            best_x = local_best_x.copy()
# EVOLVE-BLOCK-END
