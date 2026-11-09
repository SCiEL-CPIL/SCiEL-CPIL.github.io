"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal CMA-ES for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s

        rng = np.random.default_rng(42)
        n = self.dim

        # Initialize
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension step sizes (diagonal sigma)
        sigma = np.ones(n) * 1.5

        # CMA-ES parameters for diagonal variant
        mu = max(1, n // 2)  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Evolution path for sigma
        p_sigma = np.zeros(n)
        # Evolution path for covariance (diagonal)
        p_c = np.zeros(n)

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        calls = 1
        for gen in range(self.budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate lambda offspring
            lam = max(4, 2 * n)
            # But we only have budget - calls remaining, so cap lambda
            remaining = self.budget - calls
            lam = min(lam, remaining)
            if lam < 2:
                break

            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    lam = i
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                F[i] = func(y)
                Y[i] = y
                calls += 1

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Compute weighted mean of selected parents
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update evolution paths
            # p_sigma update
            h_sigma = (np.sqrt(1 - np.exp(-2 / (gen + 1) * c_sigma)) /
                       (1 - np.exp(-(gen + 1) * c_sigma))) if gen > 0 else 1.0
            h_sigma = min(h_sigma, 1.0)
            d_norm = np.sqrt(np.sum((x_new - x) ** 2 / sigma ** 2))
            if d_norm < (1.4 + 2 / (n + 1)) * chi_n * h_sigma:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x) / sigma
            else:
                p_sigma = (1 - c_sigma) * p_sigma

            # p_c update
            h_c = (np.sqrt(1 - np.exp(-2 / (gen + 1) * c_c)) /
                   (1 - np.exp(-(gen + 1) * c_c))) if gen > 0 else 1.0
            h_c = min(h_c, 1.0)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

            # Update diagonal covariance (per-dimension sigmas)
            # Use rank-1 and rank-mu updates on the diagonal
            delta_sigma = 1 + c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
            sigma *= delta_sigma

            # Rank-mu update on diagonal
            y_w = (Y_sorted[:mu] - x) / sigma
            y_w = y_w * weights[:, None]
            # Diagonal of the rank-mu matrix
            diag_mu = np.sum(y_w ** 2, axis=0)
            sigma *= np.exp(c_mu / (2 * n) * diag_mu)

            # Clamp sigmas
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Move to new mean
            x = np.clip(x_new, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            if calls >= self.budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
