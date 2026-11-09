"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+lambda)-ES with adaptive step size and restart mechanism.
    Uses a population-based approach with CMA-ES-like covariance adaptation
    for the BBOB f6 (Attractive sector) function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Track best found
        best_x = None
        best_f = np.inf

        # Total calls used
        calls_used = 0

        # We'll use a CMA-ES-like strategy
        # Population size
        lam = 4 + int(3 * np.log(self.dim))  # ~4 + 3*ln(5) ≈ 9
        mu = lam // 2

        # Initial point: try center first, then random
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        calls_used += 1
        best_x = x0.copy()
        best_f = f0

        # If we still have budget, do a CMA-ES-like search
        if calls_used < self.budget:
            # Set up CMA-ES parameters
            sigma = 2.0
            x_mean = x0.copy()
            C = np.eye(self.dim)
            # Inverse square root of C
            C_sqrt_inv = np.eye(self.dim)

            # CMA-ES parameters
            cmu = 1 / (self.dim + 4)
            c_sigma = (mu + 0.5) / (self.dim + mu + 5)
            d_sigma = 1 + 2 * max(0, (mu - 1) / (self.dim + mu - 1)) + c_sigma
            c_c = (4 + mu / self.dim) / (self.dim + 4 + 2 * mu / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu) / ((self.dim + 2) ** 2 + mu_eff))
            mu_eff = mu ** 2 / sum(1.0 / i for i in range(1, mu + 1)) if mu > 0 else 1

            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()

            while calls_used < self.budget:
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                Y = np.zeros((lam, self.dim))
                F = np.zeros(lam)

                for i in range(lam):
                    z = rng.standard_normal(self.dim)
                    y = x_mean + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5, 5)
                    Y[i] = y
                    F[i] = func(y)
                    calls_used += 1
                    if calls_used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Update mean
                x_new = sum(weights[i] * Y[i] for i in range(mu))

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ ((x_new - x_mean) / sigma)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

                # Update covariance
                delta_h = (1 + np.linalg.norm(p_c) / chi_n) * (2 - c_c) - c_c
                rank_c = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(Y[i] - x_mean, Y[i] - x_mean) for i in range(mu))
                C = (1 + c_1 + c_mu - c_c) * C + c_1 * rank_c * delta_h + c_mu * rank_mu

                # Ensure C is positive definite and symmetric
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += 1e-10 * np.eye(self.dim)

                # Update C_sqrt_inv
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                x_mean = x_new

        # Final result
        return best_x, best_f
# EVOLVE-BLOCK-END
