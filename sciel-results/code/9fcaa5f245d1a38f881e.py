"""CMA-ES based optimizer for BBOB f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES implementation for minimizing a black-box function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))  # standard lambda
        # Weighting
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Evaluate initial point
        x0 = x_mean.copy()
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0

        # Track evaluations
        eval_count = 1

        # Main loop
        while eval_count < budget:
            # Generate population
            X = np.empty((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                X[i] = x

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                eval_count += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update evolution paths
            y = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - eval_count)))) <= (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig) * C
            C += c_1 * (np.outer(pc, pc) + delta_hsig * C)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - x_mean, X_sorted[:mu] - x_mean) / sigma ** 2, axis=0)

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e5)

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
