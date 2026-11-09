"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f4 (Büche-Rastrigin) with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 300s limit, 15s safety margin
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4.0 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        c_1_mu = c_1 + c_mu

        # Initialization
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = m.copy()
        best_f = np.inf
        calls = 0
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))

        while calls + lam <= budget:
            if time.time() >= deadline:
                break

            # Sample population
            Z = np.random.standard_normal((lam, dim))
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            Y = Z @ C_sqrt.T
            X = m + sigma * Y
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam

            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            F_sorted = F[idx_sort]

            # Update mean
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)

            # Update step size (CSA)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, 1e-6)

            # Update covariance
            p_c = (1 - c_c) * p_c + c_c * delta
            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
            C_new = (C_new + C_new.T) / 2

            m = m_new
            C = C_new

        return best_x, best_f
# EVOLVE-BLOCK-END
