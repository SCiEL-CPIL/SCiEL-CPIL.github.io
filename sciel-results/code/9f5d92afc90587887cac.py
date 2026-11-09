"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES black-box optimizer for BBOB f22 (21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        alpha_mu = 0.5 * (1 + np.sum(np.arange(1, lam + 1) ** -2) / lam)
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (mu_eff + 3)) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x_mean.copy()
        best_f = np.inf
        n_calls = 0

        def eval_point(x):
            nonlocal n_calls, best_x, best_f
            n_calls += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation of mean
        eval_point(x_mean)

        for gen in range(1, 10000):
            if n_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Generate population
            D = np.linalg.cholesky(C)
            Y = np.random.randn(lam, dim)
            P = np.einsum('ij,kj->ki', D, Y.T).T  # lam x dim
            X = x_mean + sigma * P
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([eval_point(X[i]) for i in range(lam)])
            if n_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Weighted mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update covariance
            C_old = C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - c_c) * (1 - c_c) * (1 - 2 * c_c) * C_old)
            for i in range(mu):
                y_i = (X_sorted[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                D = np.linalg.cholesky(C)

            # Update path
            y_sigma = np.sqrt(mu_eff) * (D.T @ (np.linalg.inv(D) @ (x_new - x_mean))) / sigma
            pc = (1 - c_c) * pc + c_c * y_sigma
            norm_pc = np.linalg.norm(pc)
            h_sig = (norm_pc / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            ps = (1 - c_sigma) * ps + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (D.T @ (np.linalg.inv(D) @ (x_new - x_mean)) / sigma)
            norm_ps = np.linalg.norm(ps)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Update mean
            x_mean = x_new

            # Safety: keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
