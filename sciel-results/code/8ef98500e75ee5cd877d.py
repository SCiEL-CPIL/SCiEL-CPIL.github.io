"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant with adaptive population size, stall detection, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Random restarts to find a good starting point
        best_x = None
        best_fx = np.inf
        for i in range(10):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        x = best_x.copy()
        fx = best_fx

        # CMA-ES parameters
        dim = self.dim
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        c_sigma = (dim + 2) / (dim ** 2 + 4)
        mu_sigma = 0.3
        c_c = 4 / (dim + 4)
        mu_c = 0.5
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((mu_sigma - 1 + 1 / mu_sigma) / ((dim + 2) ** 2 + dim) - c_1))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_sigma - 1) / (dim + 1)) - 1) + c_sigma

        # Population size schedule
        init_lambda = 10
        min_lambda = 4

        # Track progress for adaptive population
        stall_count = 0
        max_stall = 25
        lambda_count = init_lambda

        # Precompute weights for (mu/lambda)-selection
        def get_weights(lam):
            mu = max(2, lam // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / (w @ w)
            return w, mu, mu_eff

        w, mu, mu_eff = get_weights(lambda_count)

        # Main optimization loop
        remaining_budget = self.budget - 10
        for iteration in range(remaining_budget):
            if not time_ok():
                break

            # Adaptive population size: shrink as convergence progresses
            if stall_count > 10 and lambda_count > min_lambda:
                lambda_count = max(min_lambda, lambda_count - 1)
                w, mu, mu_eff = get_weights(lambda_count)

            # Sample population
            pop = np.zeros((lambda_count, dim))
            for i in range(lambda_count):
                # Update C_sqrt occasionally for efficiency
                if iteration % 5 == 0:
                    C_sqrt = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * (C_sqrt @ z)
                np.clip(pop[i], -5, 5, out=pop[i])

            # Evaluate population
            fvals = np.array([func(p) for p in pop])

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of best mu solutions
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += w[i] * pop_sorted[i]
            new_mean /= w[:mu].sum()

            # Success/failure for stall tracking
            if fvals_sorted[0] < fx:
                stall_count = 0
            else:
                stall_count += 1

            # Stall detection: restart from a perturbation of best point
            if stall_count >= max_stall:
                stall_count = 0
                x = best_x + 0.5 * rng.standard_normal(dim)
                np.clip(x, -5, 5, out=x)
                sigma = max(sigma * 0.5, 0.1)
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                lambda_count = init_lambda
                w, mu, mu_eff = get_weights(lambda_count)
                continue

            # Step size control (1/5 success rule)
            y = (new_mean - x) / sigma
            h_sigma = (
                np.sqrt((mu_eff - 2 + 1 / mu_eff)) * (dim + 2) + 2 / (dim + 2)
            )
            p_sigma = (
                (1 - c_sigma) * p_sigma
                + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            )
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (24 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Path and covariance update
            h_c = (
                np.sqrt((mu_c - 2 + 1 / mu_c)) * (dim + 2) + 2 / (dim + 2)
            )
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            delta_h = (
                (1 - (1 - c_c) ** (iteration + 1))
                * (1 - (1 - c_c) ** (iteration + 1))
                / (1 - c_c)
                if iteration < 1000
                else 2 * c_c / (1 - c_c) - 1
            )

            # Covariance matrix update
            delta_y = y
            C = (
                (1 - c_1 - c_mu) * C
                + c_1 * (delta_h + 1) * np.outer(p_c, p_c)
                + c_mu * sum(w[i] * np.outer(pop_sorted[i] - x, pop_sorted[i] - x) / sigma ** 2 for i in range(mu))
            )

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            x = new_mean
            np.clip(x, -5, 5, out=x)
            fx = fvals_sorted[0]

        return best_x, best_fx
# EVOLVE-BLOCK-END
