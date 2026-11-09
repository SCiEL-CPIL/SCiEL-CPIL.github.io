"""Combined optimizer: robust CMA-ES (inspiration 1) with adaptive restarts and
surrogate-guided exploration (inspiration 2/current) for Rastrigin rotated f15."""
import numpy as np
import time
import math


class Optimizer:
    """Direct optimizer using CMA-ES with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # CMA-ES Parameters
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

        def run_cmaes(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_used, best_x, best_f

            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            evals_this_run = 0
            no_improve = 0
            last_best = best_f
            t_gen = 0

            while evals_this_run < max_evals and time_ok():
                t_gen += 1

                # Generate population
                pop = np.zeros((lam, dim))
                try:
                    L = np.linalg.cholesky(C)
                except:
                    C = (C + C.T) / 2
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        break

                for i in range(lam):
                    if not time_ok() or evals_this_run >= max_evals:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = mean + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate population
                fvals = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_used >= budget or not time_ok():
                        break
                    fvals[i] = evaluate(pop[i])
                    evals_this_run += 1

                if calls_used >= budget or not time_ok():
                    break

                # Sort
                order = np.argsort(fvals)
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = pop[order[i]]

                # Check for improvement
                if y[0][0] < last_best - 1e-10:
                    last_best = y[0][0]
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 15:
                    break

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

                # Update paths
                v = (new_mean - mean) / sigma

                # Compute C^{-1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v

                norm_ps = np.linalg.norm(ps)
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)

                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)

                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - mean
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)

                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma ** 2)
                C = (C + C.T) / 2

                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = float(np.clip(sigma, 0.001, 10.0))

                mean = new_mean

                if sigma < 0.001:
                    break

            return best_f

        # Phase 1: Initial random sampling to find a promising basin
        n_random = 20
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Multi-start CMA-ES
        # Strategy: Use multiple CMA-ES runs with different starting points and sigmas
        remaining_budget = budget - calls_used

        # Split budget: 70% for CMA-ES runs, 30% for final refinement
        cmaes_budget = int(remaining_budget * 0.7)
        num_cmaes_runs = 5
        evals_per_cmaes = max(10, int(cmaes_budget / num_cmaes_runs))

        for i in range(num_cmaes_runs):
            if not time_ok():
                break

            if i == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            elif i % 2 == 0:
                x0 = best_x.copy()
                sigma0 = 0.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.5

            max_evals = min(evals_per_cmaes, budget - calls_used)
            if max_evals > 0:
                run_cmaes(x0, sigma0, max_evals)

        # Phase 3: Final refinement with small sigma CMA-ES
        if time_ok() and budget - calls_used > 10:
            max_evals = min(30, budget - calls_used)
            run_cmaes(best_x.copy(), 0.05, max_evals)

        return (best_x, best_f)
