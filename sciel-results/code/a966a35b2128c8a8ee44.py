"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(dim)) for dim=5 -> 4 + 3*1.609 = 8.8 -> 8
        # Use a larger lambda for better exploration in Rastrigin
        lam = 12
        mu = lam // 2  # number of weights

        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Step size adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + c_sigma

        # Mean adaptation parameters
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
        c_mu = (2 * (mueff - 2 + w[0] / mueff)) / ((dim + 2) ** 2 + mueff)
        c_mu = min(c_mu, 2 * c_1)

        # Initial mean and sigma
        rng = np.random.default_rng(42)
        mean = np.zeros(dim)
        sigma = 2.0  # initial step size, covering a good portion of [-5,5]^5

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute weights for recombination
        weights = np.zeros(lam)
        weights[:mu] = w

        best_x = None
        best_f = np.inf
        calls = 0

        # Initial population evaluation
        for i in range(lam):
            if calls >= budget or time.time() > deadline:
                break
            z = rng.standard_normal(dim)
            # Use C_inv_sqrt * z for sampling
            y = mean + sigma * (C_inv_sqrt @ z)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        # Sort by fitness
        # We need to store all y and fy
        # Let's restructure to store all

        # Actually, let's restructure the main loop properly

        # Reset and do it properly
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = None
        best_f = np.inf
        calls = 0

        # Precompute Cholesky of C for sampling
        # We'll use a simple approach: sample z, then y = mean + sigma * C_inv_sqrt @ z
        # But updating C_inv_sqrt efficiently is complex. For dim=5, we can just use np.linalg.cholesky

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, dim))
            F = np.zeros(lam)

            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Sample from N(0, C) using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    d = L @ z
                except np.linalg.LinAlgError:
                    d = z  # fallback
                Y[i] = mean + sigma * d
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()

            if calls < lam:
                # Not enough evaluations for full population
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Compute new mean
            new_mean = np.sum(weights[:mu] * Y_sorted[:mu], axis=0)

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                C_inv_sqrt @ (new_mean - mean) / sigma
            )

            # h_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim ** 2) + 1 / (12 * dim ** 4))
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam + 1)))
                < (1.4 + 2 / (dim + 1)) * chi_n
            )

            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (
                (new_mean - mean) / sigma
            )

            # Update covariance
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C = (
                (1 - c_1 - c_mu + delta_h) * C
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
                + c_mu * np.sum(weights[:mu] * np.outer(Y_sorted[:mu] - mean, Y_sorted[:mu] - mean), axis=0)
            )

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update C_inv_sqrt
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                C_inv_sqrt = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Update step size
            sigma = sigma * np.exp(
                c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
            )

            # Update mean
            mean = new_mean

            # Clip mean to stay in bounds
            mean = np.clip(mean, -5, 5)

            # If sigma gets too small, break
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
