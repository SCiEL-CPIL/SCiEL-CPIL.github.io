"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B polish and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0

        def within_budget():
            return (time.perf_counter() - t0) < deadline

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls
            fx = func(np.asarray(x, dtype=np.float64))
            calls += 1
            return fx

        # --- Phase 1: CMA-ES ---
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, self.dim)
        fx = call_func(x)
        best_x, best_f = x.copy(), fx

        sigma = 1.5
        mu = 2
        lambda_ = 8
        weights = np.zeros(lambda_)
        for i in range(lambda_):
            weights[i] = np.log(lambda_ / 2 + 1) - np.log(i + 1)
        w_sum = weights.sum()
        weights /= w_sum
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        C = np.eye(self.dim)
        D, V = np.linalg.eigh(C)
        C_sqrt = V @ np.diag(np.sqrt(np.maximum(D, 1e-12))) @ V.T

        while calls < self.budget and within_budget():
            # Generate lambda offspring
            offspring = np.empty((lambda_, self.dim))
            f_off = np.empty(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(self.dim)
                y = x + sigma * C_sqrt @ z
                offspring[i] = np.clip(y, -5, 5)
                f_off[i] = call_func(offspring[i])
                if f_off[i] < best_f:
                    best_x, best_f = offspring[i].copy(), f_off[i]
                if calls >= self.budget:
                    break

            if calls >= self.budget:
                break

            # Sort offspring by fitness
            idx = np.argsort(f_off)
            f_off = f_off[idx]
            offspring = offspring[idx]

            # Weighted mean of mu best
            y_w = np.zeros(self.dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i]
            y_w /= np.sum(weights[:mu])

            # Update p_sigma
            d_norm = np.linalg.norm(C_sqrt_inv @ y_w)
            h_sigma = (
                np.sqrt((1 - np.exp(-2 * c_sigma * (1 - c_sigma))) * d_norm / chi_n) < 1.4 + 2 / (self.dim + 1)
            )
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ y_w

            # Update mean
            x = x + sigma * y_w

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update C
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                z_i = C_sqrt_inv @ offspring[i]
                rank_mu += weights[i] * np.outer(z_i, z_i)
            C = (1 + c_1 * (h_sigma * (1 - c_c) * (1 + c_c) - 1) - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Update C_sqrt
            D, V = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_sqrt = V @ np.diag(np.sqrt(D)) @ V.T
            C_sqrt_inv = V @ np.diag(1.0 / np.sqrt(D)) @ V.T

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Stop if sigma is too small
            if sigma < 1e-15:
                break

        # --- Phase 2: L-BFGS-B polish from best point ---
        if within_budget() and best_x is not None:
            try:
                def wrapper(x):
                    return func(np.clip(x, -5, 5))

                res = minimize(
                    wrapper,
                    best_x,
                    method="L-BFGS-B",
                    bounds=[(-5, 5)] * self.dim,
                    options={"maxiter": 100, "maxfun": 50, "ftol": 1e-15},
                )
                if res.fun < best_f:
                    best_x, best_f = np.clip(res.x, -5, 5), res.fun
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
