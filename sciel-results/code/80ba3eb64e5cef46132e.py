"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with covariance matrix adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Evaluate initial point
        x_clipped = np.clip(x, -5, 5)
        fx = func(x_clipped)
        best_x = x_clipped.copy()
        best_fx = fx
        calls_made = 1

        # Compute Cholesky of C
        C_cholesky = np.linalg.cholesky(C)

        for gen in range(budget // lambda_):
            if time.time() > deadline:
                break

            # Generate population
            Y = rng.standard_normal((lambda_, dim))
            P = Y @ C_cholesky.T
            X = x + sigma * P
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.empty(lambda_)
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                F[i] = func(X[i])
                calls_made += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = X[i].copy()

            if calls_made >= budget:
                break

            # Sort
            order = np.argsort(F)
            X_sorted = X[order]

            # Weighted mean
            m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_cholesky @ (m - x))
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = m

            # Update covariance
            pc = (1 - c_c) * pc + c_c * h_sigma * (m - x)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + C - np.eye(dim))
            C += c_mu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - x, X_sorted[:mu] - x), axis=0)
            C = (C + C.T) / 2

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, 1e10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update Cholesky
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_cholesky = np.eye(dim)

        # Final local polish
        if time.time() < deadline - 5 and calls_made < budget:
            sigma_local = 0.05
            for _ in range(min(20, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = best_x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_fx:
                    best_x = y
                    best_fx = fy
                    sigma_local *= 1.3
                else:
                    sigma_local *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
