"""Optimizer for BBOB-style f14 (Different powers) in 5D."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for 5D BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        # Phase 1: Initial random scan (20 calls)
        for _ in range(20):
            if not time_check():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with adaptive restarts
        # Budget allocation: use remaining calls for CMA-ES
        calls_used = 20
        remaining = self.budget - calls_used

        # Multiple CMA-ES runs with different initializations
        # Each run gets a portion of the remaining budget
        num_runs = 3
        calls_per_run = remaining // num_runs

        for run_idx in range(num_runs):
            if not time_check():
                break

            # Initialize CMA-ES
            if run_idx == 0:
                x0 = best_x.copy()
            else:
                # Perturb the best solution or use random
                if run_idx == 1:
                    x0 = best_x + rng.normal(0, 1.0, self.dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)

            # CMA-ES parameters for 5D
            n = self.dim
            lambda_ = 4 * int(1 + 3 * np.log(n))  # population size ~ 44 for n=5
            mu = lambda_ // 2  # selection size
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size
            sigma = 0.5 * np.linalg.norm(x0 - best_x) if run_idx > 0 else 1.0
            sigma = max(sigma, 0.1)

            # CMA-ES state
            x_mean = x0.copy()
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            # Precompute square root of C (initially identity)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            calls_in_run = 0
            local_best_x = x_mean.copy()
            local_best_f = func(x_mean)
            calls_in_run += 1
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Generate initial population
            for gen in range(calls_per_run // lambda_ + 1):
                if not time_check() or calls_in_run >= calls_per_run:
                    break

                # Generate population
                offspring = []
                offspring_f = []
                for i in range(lambda_):
                    if not time_check() or calls_in_run >= calls_per_run:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * C_sqrt @ z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_in_run += 1
                    offspring.append(y)
                    offspring_f.append(fy)
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()

                if len(offspring) < mu:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                selected = [offspring[i] for i in idx[:mu]]
                selected_f = [offspring_f[i] for i in idx[:mu]]

                # Update mean
                old_mean = x_mean.copy()
                x_mean = np.sum([weights[i] * selected[i] for i in range(mu)], axis=0)
                x_mean = np.clip(x_mean, -5, 5)

                # Update step size
                y_vec = (x_mean - old_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_in_run // lambda_ + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1) - 1:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n + 1) - 1))
                    sigma = max(sigma, 1e-8)

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1) - 1 else 0

                rank_one = np.outer(y_vec, y_vec)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    z_i = np.linalg.solve(C_sqrt, (selected[i] - old_mean) / sigma)
                    rank_mu += weights[i] * np.outer(z_i, z_i)

                C = (1 + c_1 * (delta_h - 1) + c_mu) * C + c_1 * (p_c @ p_c.T - (1 - delta_h) * C) + c_mu * (rank_mu - C)

                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update square root
                eigvals, eigvecs = np.linalg.eigh(C)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

                # Termination check
                if sigma < 1e-8 or np.std([offspring_f[i] for i in idx[:mu]]) < 1e-10:
                    break

        return best_x, best_f
