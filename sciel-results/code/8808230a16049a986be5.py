"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for rotated/shifted non-separable functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 285.0
        n = self.dim
        calls = 0

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (nu_eff + 2) / (n + nu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((nu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + nu_eff / n) / (n + 4 + 2 * nu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + nu_eff)
        c_mu = min(1 - c_1, 2 * (nu_eff - 2 + 2 / nu_eff) / ((n + 2) ** 2 + nu_eff))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Evaluate initial point
        x_eval = np.clip(x_mean, -5, 5)
        best_x = x_eval.copy()
        best_f = func(x_eval)
        calls += 1

        # Generate initial population
        for gen in range(self.budget - calls):
            if time.time() > deadline:
                break

            # Sample lambda points
            D = np.linalg.cholesky(C)
            Y = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (D @ z)
                Y.append(y)
            Y = np.array(Y)

            # Evaluate and clip
            F = []
            for y in Y:
                y_clipped = np.clip(y, -5, 5)
                f = func(y_clipped)
                F.append(f)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = y_clipped.copy()

            F = np.array(F)

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update step-size control variables
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * (D @ (x_mean_new - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update covariance matrix
            delta_hsig = 1 - hsig
            pc = (1 - c_c) * pc + delta_hsig * np.sqrt(c_c * (2 - c_c) * nu_eff) * (x_mean_new - x_mean) / sigma

            C = (1 - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig * (c_c * (2 - c_c) - 1) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x_mean, Y_sorted[:mu] - x_mean), axis=0) / sigma ** 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Update mean
            x_mean = x_mean_new

            # Recompute Cholesky decomposition
            try:
                D = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Fallback: reset covariance to identity
                C = np.eye(n)
                C_inv = np.eye(n)
                D = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)

            # Check convergence
            if np.all(np.abs(Y_sorted[:1] - Y_sorted[-1:]) < 1e-10) and gen > 10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
