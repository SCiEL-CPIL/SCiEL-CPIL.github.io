"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(dim))  # population size
        # Standard CMA-ES parameters
        alpha_mu = 0.5
        mu = lambda_ // 2  # number of best individuals
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        sigma_0 = 2.0

        def cma_es(x0, sigma, max_evals, rng):
            x = x0.copy()
            sigma = sigma
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            hs = np.sqrt((1 + c_sigma) / (1 - c_sigma)) + 1
            C = np.eye(dim)
            evals = 0
            best_x = x0.copy()
            best_f = func(x0)
            evals += 1
            while evals < max_evals:
                # Generate population
                Y = np.array([rng.standard_normal(dim) for _ in range(lambda_)])
                X = x + sigma * (Y @ (B * D))
                # Evaluate
                F = np.array([func(np.clip(xi, -5, 5)) for xi in X])
                evals += lambda_
                # Sort
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
                # Update mean
                x_new = sum(w * X[i] for i, w in enumerate(weights))
                # Update sigma
                y = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                if np.linalg.norm(p_sigma) / hs < (1.4 + 1 / dim) * np.sqrt(1 - (1 - c_sigma) ** (2 * (max_evals - evals))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                # Update C
                y_c = (x_new - x) / np.sqrt((1 - (1 - c_c) ** 2 * (max_evals - evals)))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                rank_mu = np.outer(y_c, y_c) + (c_c / (1 - c_c)) * (np.outer(p_c, p_c) - C)
                rank_one = (c_sigma / (1 - c_sigma)) * (np.outer(p_sigma, p_sigma) - C)
                C = (1 + c_c * (2 - c_c) * mu_eff - c_sigma) * C + c_sigma * rank_one + c_c * rank_mu
                # Ensure positive definiteness
                C = (C + C.T) / 2
                # Update B, D
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                x = x_new
            return best_x, best_f, evals

        # Multi-restart strategy
        rng = np.random.default_rng(12345)
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        total_evals = 0

        # Initial diverse points
        initial_points = [
            np.array([0.0] * dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-5, 5, dim),
        ]

        # Phase 1: Quick exploration with small CMA-ES runs
        for i, x0 in enumerate(initial_points[:3]):
            if time.time() > deadline - 30:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            total_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            if total_evals > budget - 50:
                break

        # Phase 2: Main CMA-ES from best point
        if time.time() < deadline - 10 and total_evals < budget - 20:
            remaining = budget - total_evals
            x0 = best_x.copy() if not np.any(np.isnan(best_x)) else np.array([0.0] * dim)
            bx, bf, used = cma_es(x0, 1.5, remaining, rng)
            total_evals += used
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Local refinement with smaller sigma
        if time.time() < deadline - 5 and total_evals < budget - 10:
            remaining = budget - total_evals
            if not np.any(np.isnan(best_x)):
                bx, bf, used = cma_es(best_x, 0.5, remaining, rng)
                total_evals += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
