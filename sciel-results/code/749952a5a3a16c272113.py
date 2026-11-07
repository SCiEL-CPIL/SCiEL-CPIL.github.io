"""Optimized BBOB-style f20 (Schwefel) minimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin from 300s limit

        dim = self.dim
        budget = self.budget

        # Initial population: uniform random points
        rng = np.random.default_rng(42)

        # Start with a reasonably sized population
        lambda_pop = max(4, dim + 2)  # standard CMA-ES pop size

        # Initial center: origin (often good for symmetric problems)
        x_mean = np.zeros(dim)

        # Initial sigma
        sigma = 3.0  # wide initial step to explore [-5,5]^5

        # CMA-ES parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = c_sigma + 2 * min(1, (mu_eff - 1) / (dim + 1)) * (1 - c_sigma)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Sort indices for weighted recombination
        # We'll evaluate lambda_pop individuals per generation

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            D = np.sqrt(np.diag(C))
            inv_D = np.where(D > 0, 1.0 / D, 0.0)
            C_inv_D = C @ (1.0 / D)[:, None]

            # Sample
            z = rng.standard_normal((lambda_pop, dim))
            x_pop = np.array([
                x_mean + sigma * (C_inv_D @ z[i]) for i in range(lambda_pop)
            ])
            # Clip to box
            x_pop = np.clip(x_pop, -5, 5)

            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_made += lambda_pop

            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()

            # Sort by fitness
            sort_idx = np.argsort(f_pop)
            x_sorted = x_pop[sort_idx]
            f_sorted = f_pop[sort_idx]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update step size control variable
            y_c = (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_c
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + lambda_pop)))) < (1.4 + 2 / (dim + 1)) * chi_n

            sigma *= np.exp(d_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update evolution path
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu * mu_eff - c_1 - delta_hsig) * C \
                + c_1 * (pc[:, None] * pc[None, :] + (1 - hsig) * delta_hsig * np.eye(dim)) \
                + c_mu * np.sum(weights[:, None, None] * (x_sorted[:mu] - x_mean)[:, None, :] * (x_sorted[:mu] - x_mean)[:, None, :], axis=0) / sigma ** 2

            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            # Clip eigenvalues to avoid numerical issues
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new_mean

            # Adaptive population size: increase if progress is good
            # Simple heuristic: if we made progress, keep lambda; otherwise reduce
            # For now, keep fixed lambda for stability

        return best_x, best_f
# EVOLVE-BLOCK-END
