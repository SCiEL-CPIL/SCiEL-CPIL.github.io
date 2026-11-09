"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop = max(4, int(4 + 3 * np.log(n)))
        c_sigma = (1 + 2 * max(0, np.sqrt((pop - 1) / (n + 1)) - 1)) / (n + 4)
        c_c = (4 + pop / n) / (n + 4 + 4 * pop / n)
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w ** 2).sum()
        c_c = (4 + pop / n) / (n + 4 + 4 * pop / n)
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff + 1) / (n + 1) - 1)) + c_sigma
        c_h = (5 + pop / n) / (n + 5)

        rng = np.random.default_rng(0)
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        hsig = 1.7

        def update_chol(C):
            nonlocal C_inv, C_sqrt, C_inv_sqrt
            C_sqrt = np.linalg.cholesky(C)
            C_inv = np.linalg.inv(C)
            # C^{-1/2} via eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

        update_chol(C)

        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        for gen in range(budget - 1):
            # Deadline check
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((pop, n))
            F = np.zeros(pop)
            for i in range(pop):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                Y[i] = x + sigma * C_sqrt @ z
                F[i] = func(Y[i])
                calls += 1

            # Sort by fitness
            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]

            # Update best
            if Fs[0] < best_f:
                best_f = Fs[0]
                best_x = Ys[0].copy()

            # Mean update
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += w[i] * (Ys[i] - x) / sigma
            x = x + sigma * y_mean

            # Cumulation
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ y_mean)
            norm_pc = np.linalg.norm(p_c)
            hsig = (norm_pc / np.sqrt(1 - (1 - c_c) ** (2 * (calls - 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            p_sigma = (1 - c_sigma) * p_sigma + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            norm_ps = np.linalg.norm(p_sigma)
            hsig2 = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (n + 1)) * chi_n

            # Covariance update
            C = (1 + c_sigma * (2 - c_sigma) * hsig2 - c_sigma) * C
            for i in range(mu):
                zi = C_inv_sqrt @ (Ys[i] - x) / sigma
                C += c_sigma * w[i] * (np.outer(zi, zi) - C)

            # Regularize C
            C = (C + C.T) / 2
            eigvals, _ = np.linalg.eigh(C)
            if eigvals.min() < 1e-10:
                C += 1e-10 * np.eye(n)

            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            update_chol(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
