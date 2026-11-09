"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 2 * int(np.floor(3 + 3 * np.log(dim)))
        mu = lam // 2

        # CMA-ES parameters
        mu_eff = 1.0 / np.sum(1.0 / np.arange(1, mu + 1))
        c_sigma = (mu_eff + 0.5) / (dim + 4 + mu_eff * 0.5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        weights = np.log(mu) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        nu = 1.0 / (1.0 - weights[0])
        chi_lam = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize at origin
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Initial function evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        while calls_made < budget:
            # Generate population
            ys = []
            zys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                ys.append(y)
                zys.append(z)

            # Evaluate all
            fys = []
            for y in ys:
                fy = func(y)
                fys.append(fy)
                calls_made += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Sort by fitness
            idx = np.argsort(fys)
            ys_sorted = [ys[i] for i in idx]
            zys_sorted = [zys[i] for i in idx]

            # Weighted mean of top mu
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (y_w - x) / sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made))) < (chi_lam + 1.4) * np.sqrt(1 + (1 - c_sigma) / c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_lam - 1))

            # Update mean
            x = y_w

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (y_w - x) / sigma

            # Update covariance matrix
            delta = np.zeros((dim, dim))
            for i in range(mu):
                diff = C_inv_sqrt @ (ys_sorted[i] - y_w)
                delta += weights[i] * np.outer(diff, diff)
            C = (1 + c_1 * (np.dot(pc, pc) - np.trace(C)) + c_mu * delta) * C

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update Cholesky and inverses
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_inv = np.linalg.inv(C)
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return best_x, best_fx
# EVOLVE-BLOCK-END
