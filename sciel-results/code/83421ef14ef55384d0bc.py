"""Rewritten optimizer using CMA-ES for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer for the sharp ridge problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        # Weights for selection
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / d) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

        # Initial point and sigma
        # For BBOB, starting at 0 is often good, but let's start near center
        x = np.zeros(d)
        sigma = 2.0  # Initial step size, box is [-5,5] so this is reasonable

        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Covariance matrix
        C = np.eye(d)
        C_eigvals = np.ones(d)
        C_eigvecs = np.eye(d)
        # Inverse square root of C for efficient updates
        C_inv_sqrt = np.eye(d)

        # Best solution
        best_x = x.copy()
        best_f = func(x)
        evals = 1

        # Precompute some constants
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        while evals < budget:
            # Update inverse square root of C
            # C = C_eigvecs @ diag(C_eigvals) @ C_eigvecs.T
            # C^{-1/2} = C_eigvecs @ diag(1/sqrt(C_eigvals)) @ C_eigvecs.T
            C_inv_sqrt = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T

            # Generate population
            ys = np.zeros((lam, d))
            for i in range(lam):
                ys[i] = C_inv_sqrt @ rng.standard_normal(d)

            # Evaluate population
            fs = np.zeros(lam)
            for i in range(lam):
                y = x + sigma * ys[i]
                y = np.clip(y, -5, 5)
                fs[i] = func(y)
                evals += 1

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = x + sigma * ys[0]
                best_x = np.clip(best_x, -5, 5)

            # Mean update
            m = sum(weights[i] * ys[i] for i in range(mu))

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ m
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals % lam))) <= (1.4 + 2 / (d + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * m
            # Note: In standard CMA-ES, the condition for h_sigma uses the norm of p_sigma
            # and the threshold depends on the expected norm.

            # Update mean
            x = x + sigma * m

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - np.eye(d))
            C += c_mu * sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Recompute eigen decomposition
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid numerical issues
            C_eigvals = np.maximum(C_eigvals, 1e-12)

        return best_x, best_f
# EVOLVE-BLOCK-END
