"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s

        # Wrap func to count calls
        call_count = [0]
        best_x = None
        best_f = np.inf

        def counting_func(x):
            nonlocal best_x, best_f
            call_count[0] += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Step size
        sigma = 2.0
        # Initial mean
        rng = np.random.default_rng(42)

        def cma_es(mu, sigma, budget_remaining, deadline):
            nonlocal call_count, best_x, best_f
            n = dim
            lam = lambda_
            mu_sel = lam // 2
            # Weights
            weights = np.log(mu_sel + 0.5) - np.log(np.arange(1, mu_sel + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Covariance matrix
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)

            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Generate initial population
            while call_count[0] < budget and time.time() < deadline:
                # Generate offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    if call_count[0] >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(n)
                    y = mu + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = counting_func(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) == 0:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu_sel):
                    y_w += weights[i] * (offspring[i] - mu) / sigma
                mu_new = mu + sigma * y_w
                mu_new = np.clip(mu_new, -5.0, 5.0)

                # Update sigma
                y_sigma = np.zeros(n)
                for i in range(mu_sel):
                    z_i = np.linalg.solve(C_sqrt, (offspring[i] - mu) / sigma)
                    y_sigma += weights[i] * z_i
                norm_y_sigma = np.linalg.norm(y_sigma)
                h_sigma = min(1.0, (norm_y_sigma / chi_n) * np.sqrt(1 - np.exp(-2 * (1 - c_sigma) * (call_count[0] / lam))))
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_y_sigma / chi_n - 1))
                sigma = min(sigma, 10.0)
                sigma = max(sigma, 1e-10)

                # Update covariance
                y_w = np.zeros(n)
                for i in range(mu_sel):
                    y_w += weights[i] * (offspring[i] - mu) / sigma
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (call_count[0] / lam)) < (1 + 2 / (n + 1)) * chi_n)
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu_sel):
                    z_i = (offspring[i] - mu) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                C = (1 + c_1 * (h_c - 1) * np.outer(p_c, p_c) + c_1 * rank_one - c_1) * C + c_mu * rank_mu
                C = (C + C.T) / 2

                # Update square roots (simple Cholesky for small n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    # Regularize
                    C += 1e-6 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                        C_inv_sqrt = np.linalg.inv(C_sqrt)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_sqrt = np.eye(n)
                        C_inv_sqrt = np.eye(n)

                mu = mu_new

                # Check for early termination
                if sigma < 1e-12 or call_count[0] >= budget:
                    break

                if time.time() >= deadline:
                    break

        # Multi-start: try a few initial points
        # First, try a grid of starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners
        for axis in range(dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[axis] = val
                start_points.append(pt)
        # Random points
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, dim))

        # Allocate budget across starts
        starts_per_budget = max(1, budget // (len(start_points) + 2))

        for start in start_points:
            if call_count[0] >= budget or time.time() >= deadline:
                break
            remaining = min(starts_per_budget, budget - call_count[0])
            if remaining < 10:
                break
            mu = start.copy()
            sigma = 1.5
            cma_es(mu, sigma, remaining, deadline)

        # If we still have budget, do a final (1+1)-ES from best point
        if call_count[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while call_count[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = counting_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
