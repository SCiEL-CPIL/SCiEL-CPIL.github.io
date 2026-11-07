"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)

        # CMA-ES parameters
        sigma = 1.0
        # Covariance matrix (start as identity)
        C = np.eye(n)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_eigvals = np.sqrt(eigvals)

        # Adaptation parameters
        cc = (4 + n / 3) / (n + 4 + 2 * n / (n + 1))
        cs = (n + 1 + 2 * int(n / 2)) / (n + 5 + 2 * int(n / 2))
        c1 = 2 / ((n + 1.3) ** 2 + n)
        cmu = min(1 - c1, 2 * (n - 2 + 1.0 / n) / ((n + 2) ** 2 + n))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        damps = 1 + 2 * max(0, np.sqrt((n + 1) / 2 - 1) - 1) + cs
        damps = 1 / (1 - cs)  # Actually, let's use standard values
        # Standard CMA-ES: damps = 1 + 2*max(0, sqrt((n+1)/2 - 1)) + cs
        damps = 1 + 2 * max(0.0, np.sqrt((n + 1) / 2.0 - 1.0)) + cs

        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        weights = np.log(np.arange(1, lam + 1))
        weights = weights / np.sum(weights)
        mueff = int(np.sum(weights[:n]) ** 2 / np.sum(weights ** 2))

        calls_used = 1
        best_x = x.copy()
        best_fx = fx

        for gen in range(budget - 1):
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform using eigenvectors and sqrt eigenvalues
                y = eigvecs @ (sqrt_eigvals * z)
                xi = x + sigma * y
                # Clip to box
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_used += 1
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()
                offspring.append((fi, xi))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * offspring[i][1] for i in range(mueff))

            # Update covariance matrix
            diff = (x - x_old) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * diff
            pc_norm_sq = np.dot(pc, pc)
            hsig = (np.sqrt(pc_norm_sq / n) / chi_n < (1.4 + 2 / (n + 1)) * (1 - cs))
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * diff

            # Rank-1 update
            delta = (1.0 if hsig else 0.0) * (1 - c1 - cmu) * np.outer(pc, pc)
            C = (1 + c1 + cmu - c1) * C + c1 * np.outer(diff, diff) + cmu * sum(
                weights[i] * np.outer((offspring[i][1] - x_old) / sigma, (offspring[i][1] - x_old) / sigma)
                for i in range(mueff)
            ) - delta

            # Ensure positive semi-definite
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            eigvals_new = np.maximum(eigvals_new, 1e-12)
            C = (eigvecs_new * eigvals_new) @ eigvecs_new.T

            # Update step size (sigma)
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)

            # Stop if we've used all budget
            if calls_used >= budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
