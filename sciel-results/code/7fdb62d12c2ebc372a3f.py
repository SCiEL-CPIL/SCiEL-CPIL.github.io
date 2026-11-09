"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, cumulative step-size control, starting at origin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lam = 4  # population size
        mu = lam // 2  # number of parents

        # CMA-ES parameters
        weights = np.ones(mu)
        weights = weights / weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / (dim + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_sigma = c_sigma / d_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff)

        # Initialize
        x = np.zeros(dim)  # start at origin
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Evaluate initial point
        x = np.clip(x, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_used = 1

        # Pre-compute
        w = weights
        w_sqrt = np.sqrt(w)

        for gen in range(budget // lam):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                Y[i] = x + sigma * z

            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                calls_used += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()

            # Check deadline after each generation
            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Recombination
            y_mean = np.sum(w[:, None] * Y[:mu], axis=0)
            d = y_mean - x
            x = y_mean

            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (sigma / chi_n) * (C_sqrt_inv @ d)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update evolution path for C
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * d / sigma

            # Update covariance matrix
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(Y[i] - x, Y[i] - x)

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
