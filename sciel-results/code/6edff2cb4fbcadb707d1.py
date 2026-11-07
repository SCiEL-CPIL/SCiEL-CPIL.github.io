"""Optimized BBOB solver using CMA-ES with restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Multi-start CMA-ES
        # Use multiple restarts with different initializations
        n_restarts = 6
        budget_per_restart = max(30, budget // n_restarts)
        # Last restart gets remaining budget
        budgets = [budget_per_restart] * (n_restarts - 1)
        budgets.append(budget - budget_per_restart * (n_restarts - 1))

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if budget - r * budget_per_restart - (budgets[r] if r < len(budgets) else 0) < 10:
                break

            # Different initialization for each restart
            if r == 0:
                x0 = np.zeros(dim)
            elif r == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif r == 2:
                x0 = rng.uniform(-4, 4, dim)
            elif r == 3:
                x0 = rng.uniform(-5, 5, dim)
            elif r == 4:
                # Bias towards corners
                x0 = rng.choice([-4, 4], dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            update_best(x0, f0)

            # CMA-ES parameters
            sigma = 1.0
            lambda_pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            c_mu = weights[0]
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = (2 * (c_mu - 0.5 + c_mu * (1 - c_1 - c_c))) / ((dim + 2) ** 2 + c_mu)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_c
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            inv_sqrt_C = np.eye(dim)  # Will be updated
            # Precompute for efficiency
            ps = 0.0

            # Evaluate initial population
            pop = []
            for i in range(lambda_pop):
                xi = x0 + sigma * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                pop.append((xi, fi))
                update_best(xi, fi)

            # Sort by fitness
            pop.sort(key=lambda t: t[1])

            # CMA-ES iterations
            for it in range(budgets[r] - lambda_pop):
                if time.time() > deadline:
                    return best_x, best_f

                # Update mean
                x_mean = sum(weights[i] * pop[i][0] for i in range(mu))

                # Update evolution paths
                d = x_mean - x0
                x0 = x_mean

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * inv_sqrt_C @ d
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)

                # Update correlation matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d
                delta_heaviside = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                C = (1 + c_1 * delta_heaviside - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(pop[i][0] - x_mean, pop[i][0] - x_mean) for i in range(mu))

                # Ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update inverse square root
                eigvals_sqrt = np.sqrt(eigvals)
                inv_sqrt_C = eigvecs @ np.diag(1 / eigvals_sqrt) @ eigvecs.T

                # Generate new population
                new_pop = []
                for i in range(lambda_pop):
                    xi = x_mean + sigma * inv_sqrt_C @ rng.standard_normal(dim)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    new_pop.append((xi, fi))
                    update_best(xi, fi)

                new_pop.sort(key=lambda t: t[1])
                pop = new_pop

                mu_eff = 1 / sum(weights[:mu] ** 2)

        # Phase 2: Local polish around the best point
        if time.time() < deadline - 5:
            if not np.isnan(best_f):
                # Fine-tune with small step size
                x_curr = best_x.copy()
                f_curr = best_f
                sigma = 0.1
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < f_curr:
                        x_curr, f_curr = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.7
                    update_best(x_curr, f_curr)

        return best_x, best_f
# EVOLVE-BLOCK-END
