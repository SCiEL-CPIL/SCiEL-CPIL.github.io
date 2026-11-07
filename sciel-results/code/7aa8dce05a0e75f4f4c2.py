"""Rewritten program for the suite task; new approach: CMA-ES with adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / n) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial parameters
        x0 = np.zeros(n)
        sigma = 2.0
        B = np.eye(n)
        D = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Precompute sqrtm for C (using eigen decomposition for symmetry)
        def update_C_sqrt():
            nonlocal B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            return B, D

        B, D = update_C_sqrt()

        for it in range(budget - 1):
            # Generate population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            for i in range(lam):
                z = np.random.standard_normal(n)
                y = x0 + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Mean update
            w = weights[:mu]
            ymean = np.sum(w[:, None] * ys[:mu], axis=0)
            x0_new = x0 + sigma * ymean

            # Step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * ymean))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 1 / n) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Mean update with cumulative step
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * ymean
            delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1.4 + 1 / n) * chi_n else 0

            # Covariance matrix update
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            for i in range(mu):
                zi = ys[i]
                C += c_mu * w[i] * np.outer(zi, zi)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = (eigvecs * eigvals) @ eigvecs.T

            x0 = x0_new
            B, D = update_C_sqrt()

        return best_x, best_f
# EVOLVE-BLOCK-END
