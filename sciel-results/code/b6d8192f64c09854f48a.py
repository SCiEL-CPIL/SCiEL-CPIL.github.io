"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(n))  # population size
        mu = lambda_ // 2  # number of selected individuals

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x0 = np.zeros(n)
        sigma = 2.0

        # Means and covariance
        mean = x0.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Helper: compute inverse square root of covariance matrix
        def _compute_cmat_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ (eigvals_inv_sqrt[:, None] * eigvecs.T)
            C_inv = eigvecs @ (eigvals[:, None] * eigvecs.T)
            return C_inv, C_inv_sqrt

        # Initial decomposition
        C_inv, C_inv_sqrt = _compute_cmat_inv_sqrt(C)

        def _update_decomposition(mean, C):
            nonlocal C_inv, C_inv_sqrt
            C_inv, C_inv_sqrt = _compute_cmat_inv_sqrt(C)

        # Main loop
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1

        for gen in range(budget):
            if time.time() > time_limit:
                break

            # Generate population
            pop = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                pop[i] = mean + sigma * (C_inv_sqrt.T @ z)
                np.clip(pop[i], -5, 5, out=pop[i])

            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls += lambda_

            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = pop[order]
            f_sorted = fvals[order]

            # Weighted mean of selected
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y_w - mean) / sigma)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - mean) / sigma

            # Update mean
            mean = y_w

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y_sorted[:mu] - mean, y_sorted[:mu] - mean), axis=0)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-18)
            C = eigvecs @ (eigvals[:, None] * eigvecs.T)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update decomposition
            _update_decomposition(mean, C)

            # Early stopping if sigma is too small or too large
            if sigma < 1e-12 or sigma > 1e10:
                break

            # Check if we've used all budget
            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
