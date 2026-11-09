"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time and budget guards for BBOB optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        # CMA-ES parameters for n=5
        lam = 10  # Population size
        mu = lam // 2  # Number of selected points
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Initial evaluation
        x_init = np.clip(x_mean, -5, 5)
        f_best = func(x_init)
        x_best = x_init.copy()
        eval_count = 1

        gen = 0
        while eval_count < budget:
            # Time check
            if time.time() - start_time > time_limit:
                break

            gen += 1

            # Generate population
            Y = np.empty((lam, n))
            F = np.empty(lam)

            for i in range(lam):
                if eval_count >= budget:
                    break
                if time.time() - start_time > time_limit:
                    break

                try:
                    z = np.random.standard_normal(n)
                    y = x_mean + sigma * C_sqrt @ z
                    y_clipped = np.clip(y, -5, 5)
                    f_val = func(y_clipped)
                except Exception:
                    f_val = np.inf
                    y_clipped = x_best.copy()

                Y[i] = y_clipped
                F[i] = f_val
                eval_count += 1

                if f_val < f_best:
                    f_best = f_val
                    x_best = y_clipped.copy()

            if eval_count >= budget:
                break
            if time.time() - start_time > time_limit:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_sel = Y[order[:mu]]
            F_sel = F[order[:mu]]

            # Update mean
            x_old_mean = x_mean.copy()
            x_mean = np.sum(w[:, None] * Y_sel, axis=0)

            # Update p_sigma
            norm_c = np.sqrt(np.trace(C))
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (C_sqrt.T @ ((x_mean - x_old_mean) / sigma)) / norm_c

            # h_sigma
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < threshold else 0.0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old_mean) / sigma

            # Update C
            delta_h = (2 - h_sigma) * c_c * (1 - c_c) - c_1 * (1 - (1 - c_c)**(2 * (gen + 1)))
            rank_one = p_c @ p_c + (1 - h_sigma) * delta_h * sigma * sigma * C
            rank_mu = np.sum(w[:, None] * (Y_sel - x_old_mean) @ (Y_sel - x_old_mean).T, axis=0) / (sigma * sigma)
            C += -c_1 * C + c_1 * rank_one + c_mu * rank_mu

            # Update C_sqrt via Cholesky
            try:
                C_sqrt, _ = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                try:
                    C_sqrt, _ = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except:
                    C_sqrt = np.eye(n)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 100)

            if sigma < 1e-10:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
