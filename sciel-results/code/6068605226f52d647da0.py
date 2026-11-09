"""Combined optimizer: Adaptive-restart CMA-ES with Cholesky sampling for Rastrigin rotated."""
import numpy as np
import time
import math


class Optimizer:
    """Adaptive-restart CMA-ES with Cholesky sampling and decreasing-noise restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse random initialization
        n_init = min(25, budget // 8)
        init_points = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            init_points.append((fx, x))

        if best_x is None:
            return (np.zeros(dim), 0.0)

        init_points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES with adaptive restarts
        def run_cma_es_with_restarts(start_x, start_f, sigma_init, remaining_budget, restart_idx=0):
            nonlocal calls_used, best_f, best_x, deadline

            if remaining_budget <= 0 or time.time() > deadline:
                return start_x, start_f

            # CMA-ES state
            x_mean = start_x.copy()
            sigma = sigma_init
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Try Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

            calls_local = 0
            stagnation_counter = 0
            last_best_f = start_f
            max_stagnation = 8 + restart_idx * 2  # allow more stagnation on later restarts

            while calls_local < remaining_budget:
                if time.time() > deadline:
                    break

                # Generate population
                pop = []
                fvals = []
                for i in range(lam):
                    if calls_local >= remaining_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_new = x_mean + sigma * (L @ z)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    fx_new = evaluate(x_new)
                    pop.append(x_new)
                    fvals.append(fx_new)
                    calls_local += 1

                if not pop:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                sorted_pop = [pop[i] for i in order]
                sorted_fvals = [fvals[i] for i in order]

                # Check for progress
                if sorted_fvals[0] < last_best_f - 1e-10:
                    stagnation_counter = 0
                    last_best_f = sorted_fvals[0]
                else:
                    stagnation_counter += 1

                # Weighted recombination
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (sorted_pop[i] - x_mean) / sigma

                # Update paths
                v = (np.sum(weights[:, np.newaxis] * np.array(sorted_pop[:mu]), axis=0) - x_mean) / sigma

                # Compute C^{-1/2} for path update
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * (C_inv_sqrt @ v)
                norm_ps = np.linalg.norm(ps)
                t_gen = calls_local / lam
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

                # Update mean
                x_mean = x_mean + sigma * y_w

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)

                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = (sorted_pop[i] - x_mean + sigma * y_w) / sigma
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)

                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu
                C = (C + C.T) / 2

                # Ensure positive definite
                eigvals_new, eigvecs_new = np.linalg.eigh(C)
                eigvals_new = np.maximum(eigvals_new, 1e-20)
                C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T

                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = (C + C.T) / 2
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                # Step size adaptation
                sigma *= math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Restart condition: if stagnation exceeds threshold and we have budget left
                if stagnation_counter > max_stagnation and calls_local < remaining_budget * 0.7:
                    # Perturb the global best with decreasing noise
                    noise_scale = 0.5 / (restart_idx + 1)
                    new_start = np.clip(best_x + noise_scale * rng.standard_normal(dim), -5.0, 5.0)
                    new_sigma = max(0.3, sigma_init / (restart_idx + 1))
                    remaining = remaining_budget - calls_local
                    return run_cma_es_with_restarts(new_start, best_f, new_sigma, remaining, restart_idx + 1)

            return x_mean, best_f

        # Run CMA-ES from best initial point
        cma_budget = budget - calls_used - 5
        cma_budget = max(cma_budget, 30)

        _, _ = run_cma_es_with_restarts(best_x.copy(), best_f, 1.5, cma_budget, 0)

        return (best_x, best_f)
