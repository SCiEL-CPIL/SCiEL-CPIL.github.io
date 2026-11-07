"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for 5D BBOB f6 with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_used = 1

        # CMA-ES parameters
        n = dim
        lam = 4 * n  # 16
        mu = lam // 2  # 8

        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights[0] + weights.sum() ** 2 / (2 * mu))

        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def _mat_sqrt(A):
            w, V = np.linalg.eigh(A)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(np.sqrt(w)) @ V.T

        def _mat_inv_sqrt(A):
            w, V = np.linalg.eigh(A)
            w = np.clip(w, 1e-12, None)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

        C_sqrt = _mat_sqrt(C)
        C_sqrt_inv = _mat_inv_sqrt(C)

        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate and evaluate offspring
            ys = np.empty((lam, n))
            fs = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * (C_sqrt @ z), -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_used += 1
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = y.copy()

            # Sort
            idx = np.argsort(fs)
            ys_s = ys[:len(idx)][idx]
            fs_s = fs[idx]

            # New mean
            x_new = np.sum(weights[:, None] * ys_s[:mu], axis=0)

            # Sigma path
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x_new - x) / sigma)

            # h_sigma
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < (1.4 + 2.0 / (n + 1))

            # C path
            if h_sigma:
                p_c = (1 - c_c) * p_c + \
                      np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

            # Covariance update
            delta_h = (1.0 if h_sigma else 0.0) - 1.0
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                d = C_sqrt_inv @ (ys_s[i] - x) / sigma
                rank_mu += weights[i] * np.outer(d, d)

            C = ((1 + c_c * (2 - c_c) * delta_h - c_c) * C +
                 (c_c * (2 - c_c) * (1 - delta_h) / n) * np.outer(p_c, p_c) +
                 (1 - c_c) * C +
                 (1 - (1 - c_c) ** 2) * rank_mu / (1 - (1 - c_c)))

            C = (C + C.T) / 2

            C_sqrt = _mat_sqrt(C)
            C_sqrt_inv = _mat_inv_sqrt(C)

            # Sigma update
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
