"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        rng = np.random.default_rng(0)

        # Global best tracking
        global_best_x = np.zeros(self.dim)
        global_best_f = float('inf')

        def update_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # Helper: run CMA-ES from a given start point
        def cma_es(x0, budget_remaining, sigma0):
            nonlocal global_best_x, global_best_f
            dim = self.dim
            x = x0.copy()
            fx = func(x)
            update_best(x, fx)
            budget_remaining -= 1

            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            step_size = 1.0
            sigma = sigma0

            # Covariance matrix
            C = np.eye(dim)
            # Step-size adaptation
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            c_sigma = (pop_size + 2) / (4 * dim + pop_size)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size + 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + pop_size / dim) / (4 * dim + pop_size)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            h_sigma = 0.0

            while budget_remaining > 0 and time.time() < deadline:
                # Generate population
                ys = []
                for i in range(pop_size):
                    if time.time() >= deadline:
                        break
                    y = x + sigma * (C @ rng.standard_normal(dim))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    budget_remaining -= 1
                    ys.append((y, fy))

                if len(ys) == 0:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[1])

                # Update best
                for y, fy in ys:
                    update_best(y, fy)

                # Selection
                lam = len(ys)
                mu = max(1, lam // 2)
                w = np.log(mu + 1) - np.log(1 + np.arange(mu))
                w /= w.sum()
                mu_eff = 1 / (w @ w)

                # Weighted mean
                y_mean = sum(w[i] * ys[i][0] for i in range(mu))

                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff / dim) * (C @ ((y_mean - x) / sigma))
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1) * C + c_1 * (p_c @ p_c - C)
                C += c_mu * sum(w[i] * np.outer(ys[i][0] - y_mean, ys[i][0] - y_mean) for i in range(mu))

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update mean
                x = y_mean

                # Stopping criterion
                if np.max(np.diag(C)) / np.min(np.diag(C)) > 1e14:
                    break
                if sigma < 1e-10:
                    break

            return global_best_x, global_best_f

        # Multi-restart strategy
        # Start with a few diverse initial points
        n_restarts = 3
        restart_budget = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() >= deadline:
                break

            if r == 0:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 2.0
            else:
                # Perturb around previous best
                x0 = global_best_x + rng.standard_normal(self.dim) * 1.0
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            remaining = self.budget - (r * restart_budget)
            if r == n_restarts - 1:
                remaining = self.budget - (r * restart_budget)
            else:
                remaining = restart_budget

            cma_es(x0, remaining, sigma0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
