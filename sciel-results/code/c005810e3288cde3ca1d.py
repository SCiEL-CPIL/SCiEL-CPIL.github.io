"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with diverse initial points for Gallagher f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Multi-start CMA-ES: run several short CMA-ES instances from different starts
        # For 500 budget and dim=5, we can afford multiple restarts
        n_starts = 10
        budget_per_start = max(30, budget // n_starts)

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Generate diverse starting points
        start_points = []
        # Include origin and random points
        start_points.append(np.zeros(dim))
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for i, x0 in enumerate(start_points):
            # Check time
            if time.time() > deadline - 2.0:
                break

            # CMA-ES parameters
            x = x0.copy()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma: start larger, shrink over time
            sigma = 3.0
            # CMA-ES state
            C = np.eye(dim)
            C_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            cs = (dim + 2) / (dim + 8)
            ds = 1 + 2 * max(0, (np.sqrt((dim + 1) / (dim - 1)) - 1)) + cs
            cmu = min(0.5, 2 / ((dim + 4) ** 2 - 2))
            mu = dim // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            calls_in_this_start = 0
            max_calls_this_start = budget_per_start

            while calls_in_this_start < max_calls_this_start:
                # Check time
                if time.time() > deadline - 2.0:
                    break

                # Generate lambda = 2*mu candidates
                lam = 2 * mu
                offspring = []
                for k in range(lam):
                    y = x + sigma * (np.linalg.solve(np.sqrt(C), rng.standard_normal(dim)))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = [func(y) for y in offspring]
                calls_in_this_start += lam

                # Sort by fitness
                idx = np.argsort(offspring_f)
                y_sorted = [offspring[i] for i in idx]
                f_sorted = [offspring_f[i] for i in idx]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = y_sorted[0].copy()

                # Update mean
                y_mean = np.sum(np.array([weights[k] * y_sorted[k] for k in range(mu)]), axis=0)
                y_diff = y_mean - x
                x = y_mean

                # Update sigma
                p_sigma = (1 - cs) * p_sigma + cs * np.sqrt(mueff) * np.linalg.solve(np.sqrt(C), y_diff / sigma)
                sigma *= np.exp((cs / ds) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_in_this_start + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                p_c = (1 - cmu) * p_c + h_sigma * cmu * y_diff / sigma
                rank_mu = np.zeros((dim, dim))
                for k in range(mu):
                    diff = np.linalg.solve(np.sqrt(C), (y_sorted[k] - x) / sigma)
                    rank_mu += np.outer(weights[k] * diff, diff)
                C = (1 + cmu * (mueff - 1) * (1 - h_sigma) * cmu) * C + cmu * (mueff * (1 - h_sigma) * cmu * (p_c @ p_c - C) + rank_mu)
                # Ensure positive definite
                C = (C + C.T) / 2
                # Regularize
                C += 1e-12 * np.eye(dim)
                C_inv = np.linalg.inv(C)
                C = np.linalg.cholesky(C) @ np.linalg.cholesky(C).T
                C = (C + C.T) / 2

                # If no improvement in a while, restart from a new point
                if calls_in_this_start % (2 * lam) == 0 and calls_in_this_start > 4 * lam:
                    # Check if we've made progress
                    if f_sorted[0] >= fx - 1e-6:
                        break

        # Use remaining budget for a final (1+1)-ES from the best point
        remaining = budget - sum(budget_per_start for _ in range(n_starts))
        if remaining > 0 and best_x is not None and time.time() < deadline - 1.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(remaining, 50)):
                if time.time() > deadline - 0.5:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
