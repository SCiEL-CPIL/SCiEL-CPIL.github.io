"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        rng = np.random.default_rng(42)
        n = self.dim

        # Initial point
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.0
        # Population size
        lambda_ = 4 + int(3 * np.log(n))  # standard lambda
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size adaptation
        cs = (mueff + 1) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        # Covariance adaptation
        c1 = (2 * (mueff - 2 + 2 / n)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Damping for step size
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        C_sqrt = np.eye(n)
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        # Precompute
        def update_C_sqrt(C):
            D, V = np.linalg.eigh(C)
            D = np.clip(D, 1e-12, None)
            return V @ np.diag(np.sqrt(D)) @ V.T, V @ np.diag(1.0 / np.sqrt(D)) @ V.T

        C_sqrt, C_inv_sqrt = update_C_sqrt(C)

        call_count = 1
        evaluations_left = self.budget - 1

        while evaluations_left > 0:
            if time.time() > deadline - 5.0:
                break

            # Generate lambda_ offspring
            offspring = np.zeros((lambda_, n))
            f_offspring = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring[i] = y
                f_offspring[i] = func(y)
                call_count += 1
                evaluations_left -= 1
                if f_offspring[i] < best_fx:
                    best_fx = f_offspring[i]
                    best_x = y.copy()
                if evaluations_left <= 0:
                    break

            if evaluations_left <= 0:
                break

            # Sort
            idx = np.argsort(f_offspring)
            f_sorted = f_offspring[idx]
            y_sorted = offspring[idx]

            # Update mean
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            d = y_w - x
            x = y_w

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (sigma * d))
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * evaluations_left))) < (1.4 + 2 / (n + 1)) * chi_n

            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * d / sigma
            hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * evaluations_left))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update covariance
            delta_hsig = (1 - hsig) * hsig * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(ps, ps) + (1 - hsig) * delta_hsig * C) \
                + cmu * np.sum(weights[:, None, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0) / sigma ** 2

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt, C_inv_sqrt = update_C_sqrt(C)
            except:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                C_inv_sqrt = np.eye(n)

            # Step size adaptation
            sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

        # Final polish with small steps
        if time.time() < deadline - 2.0 and evaluations_left > 0:
            sigma_polish = max(sigma * 0.1, 1e-6)
            for _ in range(min(evaluations_left, 100)):
                if time.time() > deadline - 1.0:
                    break
                y = np.clip(best_x + sigma_polish * rng.standard_normal(n), -5, 5)
                fy = func(y)
                call_count += 1
                evaluations_left -= 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    sigma_polish *= 1.3
                else:
                    sigma_polish *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
