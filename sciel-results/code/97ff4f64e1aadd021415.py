"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lam // 2
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1.3)) / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x = rng.uniform(-3, 3, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 1.5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        h_sigma = 0.0

        calls_made = 1
        remaining = self.budget - 1

        while remaining > 0:
            if time.time() > deadline:
                break

            # Generate population
            population = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                population[i] = y
                fvals[i] = fy
                calls_made += 1
                remaining -= 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if remaining <= 0:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = population[order]
            f_sorted = fvals[order]

            # Update mean (weighted recombination)
            x_new = np.sum(w[:, None] * y_sorted[:mu], axis=0)

            # Update step size control
            y_sigma = (np.sqrt(np.sum((C @ C) * (C @ C))) / sigma) * (x_new - x) if np.allclose(C, np.eye(n)) else (x_new - x) / sigma
            # Simplified: use the actual step
            y_sigma = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma

            # Hessian of the norm
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update covariance control
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

            # Rank-one and rank-mu updates
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(n)) \
                + c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0)

            # Ensure positive semi-definite (symmetrize and fix)
            C = (C + C.T) / 2
            # Eigenvalue repair
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
