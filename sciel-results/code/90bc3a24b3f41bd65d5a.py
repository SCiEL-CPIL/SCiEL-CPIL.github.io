"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for minimizing func on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)

        # Strategy Parameters
        c_sigma = (mueff + 2) / (lam + mueff + 5)
        c_c = (4 + mueff / lam) / (lam + 4 + 2 * mueff / lam)
        c_1 = 2 / ((lam + 2) ** 2 + mueff)
        c_mu = min(c_1, mueff / (2 * ((lam + 2) ** 2)))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        def sample():
            return x_mean + sigma * (B @ (D * rng.standard_normal(dim)))

        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1

        while evals + lam <= budget:
            if time.time() > deadline:
                break

            # Sample population
            Y = np.array([sample() for _ in range(lam)])
            F = np.array([func(y) for y in Y])
            evals += lam

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Update mean
            x_new = np.sum(w[:, None] * Y[:mu], axis=0)

            # Update sigma
            # C^{-1/2} = B diag(1/D) B^T
            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            # Hessian damping
            if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (evals // lam))) < (1.4 + 2 / (dim + 1)) * chi_n:
                delta_sigma = 1 - c_c
            else:
                delta_sigma = 1

            C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(w[:, None] * np.outer((Y[:mu] - x_mean) / sigma, (Y[:mu] - x_mean) / sigma), axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            B = eigvecs
            D = np.sqrt(eigvals)

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
