"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with multiple restarts and time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = float('inf')

        def check_time():
            return time.time() < deadline

        def time_remaining():
            return max(0.0, deadline - time.time())

        # Multiple restart strategy
        # Use a fraction of budget per restart, with the last restart getting the remainder
        n_restarts = 3
        # Allocate budget: first 2 restarts get 30% each, last gets 40%
        restart_budgets = [int(budget * 0.3), int(budget * 0.3), budget - 2 * int(budget * 0.3)]
        
        for r in range(n_restarts):
            if calls_used >= budget:
                break
            if not check_time():
                break

            rb = restart_budgets[r]
            if rb < 10:
                rb = 10

            # Initial point for this restart
            if r == 0:
                x_init = np.zeros(dim)
            else:
                # Diversify starting points for subsequent restarts
                x_init = rng.uniform(-4, 4, dim)

            fx = func(x_init)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_init.copy()

            # CMA-ES parameters
            n = dim
            pop_size = max(4, int(4 + 3 * np.log(n)))
            # Ensure we don't exceed remaining budget
            max_iter = (rb - 1) // pop_size
            if max_iter < 5:
                # Fall back to (1+1)-ES for small budgets
                sigma = 1.5
                x = x_init.copy()
                fx_cur = fx
                while calls_used < budget + 1 and check_time():
                    y = x + sigma * rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx_cur:
                        x = y
                        fx_cur = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                continue

            sigma = 2.0
            x_mean = x_init.copy()
            step_sizes = np.ones(n)

            # CMA-ES parameters
            cs = (pop_size + 2) / (n + pop_size + 4)
            ds = 1 + 2 * max(0, (np.sqrt((pop_size + 1) / (n + 1)) - 1) / (1 + cs))
            c1 = (2) / ((n + 1.3) ** 2 + pop_size)
            cmu = min(1 - c1, 2 * ((pop_size - 2 + 1. / pop_size) / (n + 2)) ** 2 - 1)

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            weights = np.log(pop_size / 2) - np.log(np.arange(1, pop_size + 1))
            weights /= weights.sum()
            mu_eff = 1. / np.sum(weights[:pop_size // 2] ** 2)

            chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1 / (21 * n ** 2))

            for it in range(max_iter):
                if calls_used >= budget:
                    break
                if not check_time():
                    break
                # Check if enough time for one generation
                if time_remaining() < 0.1:
                    break

                # Generate offspring
                y = rng.standard_normal((pop_size, n))
                x_offspring = x_mean + sigma * step_sizes * y

                f_offspring = []
                for i in range(pop_size):
                    if calls_used >= budget:
                        break
                    if not check_time():
                        break
                    x_i = np.clip(x_offspring[i], -5, 5)
                    f_i = func(x_i)
                    calls_used += 1
                    f_offspring.append(f_i)
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()

                if len(f_offspring) < pop_size:
                    break

                f_offspring = np.array(f_offspring)
                order = np.argsort(f_offspring)
                x_sorted = x_offspring[order]
                f_sorted = f_offspring[order]

                x_mean_new = np.sum(weights[:, np.newaxis] * x_sorted, axis=0)

                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (x_mean_new - x_mean) / (sigma * step_sizes)

                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - pop_size) / pop_size)) < (1.4 + 2 / (n + 1)) * chi_n

                # Update p_c
                p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_mean_new - x_mean) / (sigma * step_sizes)

                # Update step sizes
                C_diag = step_sizes ** 2
                delta_h = (1 - h_sigma) * cs * (2 - cs)
                rank_one = np.outer(p_c, p_c)

                diffs = (x_sorted - x_mean) / (sigma * step_sizes)
                rank_mu = np.sum(weights[:, np.newaxis] * diffs ** 2, axis=0)

                C_diag = (1 + c1 * delta_h - c1 - cmu) * C_diag + c1 * rank_one + cmu * rank_mu
                C_diag = np.maximum(C_diag, 1e-10)
                step_sizes = np.sqrt(C_diag)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))

                x_mean = x_mean_new

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
