"""Optimizer for BBOB f22 using CMA-ES with restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        def cma_es_run(x0, f0, sigma0, budget, rng, func, deadline):
            """Run CMA-ES from given initial point."""
            dim = len(x0)
            x = x0.copy()
            fx = f0
            sigma = sigma0

            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2

            # Weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            # Step size control parameters
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            c_sigma = 1.0 / (ds * dim)

            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Evaluate initial population
            best_x = x.copy()
            best_f = fx
            calls = 1

            for gen in range(budget // lambda_ + 1):
                # Check deadline
                if time.time() > deadline - 5:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    fy = func(y)
                    calls += 1
                    offspring_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in idx]
                offspring_f_sorted = [offspring_f[i] for i in idx]

                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * offspring_sorted[i] for i in range(mu))

                # Update step size
                z = (x - x_old) / (sigma * np.sqrt(np.diag(C_inv_sqrt @ C_inv_sqrt.T)))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ z
                norm_ps = np.linalg.norm(ps)
                h_sig = 1 if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / lambda_)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                sigma *= np.exp(cs / c_sigma * (norm_ps / chi_n - 1))

                # Update covariance
                delta_h = (1 - h_sig) * c_c * (2 - c_c)
                pc = (1 - c_c) * pc + delta_h * (x - x_old) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sig) * delta_h * (2 - c_c) * C)
                for i in range(mu):
                    y_i = (offspring_sorted[i] - x_old) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Update C_inv_sqrt (using Cholesky)
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    C_inv_sqrt = np.linalg.inv(L)
                except:
                    C_inv_sqrt = np.eye(dim)

                if calls >= budget:
                    break

            return best_x, best_f, calls

        # Multi-restart strategy
        # First run: standard CMA-ES
        x0 = np.zeros(dim)
        f0 = func(x0)
        total_calls = 1

        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # Try different starting points
        start_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
        ]

        sigma0 = 2.0
        budget_per_run = total_budget // 3

        for sp in start_points:
            if time.time() > deadline - 10:
                break

            remaining = total_budget - total_calls
            if remaining <= 0:
                break

            run_budget = min(budget_per_run, remaining)
            if run_budget < 10:
                break

            x0 = np.clip(sp, -5, 5)
            f0 = func(x0)
            total_calls += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            bx, bf, calls_used = cma_es_run(x0, f0, sigma0, run_budget, rng, func, deadline)
            total_calls += calls_used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # If we still have budget, do a local refinement around best
        if total_calls < total_budget and time.time() < deadline - 10:
            remaining = total_budget - total_calls
            if remaining > 20:
                # Fine-tuning with small sigma
                x_curr = best_x.copy()
                f_curr = best_f
                sigma = 0.1
                for _ in range(remaining):
                    if time.time() > deadline - 5:
                        break
                    y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < f_curr:
                        x_curr = y
                        f_curr = fy
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
