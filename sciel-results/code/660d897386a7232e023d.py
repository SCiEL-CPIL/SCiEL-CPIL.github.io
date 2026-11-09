"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with adaptive covariance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        # Standard CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        sigma = 2.0
        c_sigma = (pop_size + 2) / (pop_size + n + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / n) / (n + 4 + 2 * pop_size / n)
        c_1 = 2 / ((n + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - n - 2) / ((n + 2) ** 2 + pop_size))

        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        mean = x.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)

        # Precompute for Cholesky decomposition updates
        # We'll use a simpler approach: recompute Cholesky each iteration for stability
        # For n=5 this is fast enough

        def cholesky_inv_sqrt(C):
            """Compute inverse square root of C using eigenvalue decomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        C_inv_sqrt = cholesky_inv_sqrt(C)

        ps = np.zeros(n)
        pc = np.zeros(n)

        calls_made = 1
        budget_remaining = self.budget - 1

        while budget_remaining > 0:
            if time.time() >= deadline:
                break

            # Sample population
            Y = np.zeros((pop_size, n))
            F = np.zeros(pop_size)
            for i in range(pop_size):
                z = rng.standard_normal(n)
                y = mean + sigma * C_inv_sqrt @ z
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls_made += 1
                budget_remaining -= 1
                if budget_remaining <= 0:
                    break

            if budget_remaining <= 0:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update best
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = Y[0].copy()

            # Update mean
            x_new = np.zeros(n)
            for i in range(min(mu, pop_size)):
                x_new += weights[i] * Y[i]
            mean = x_new

            # Update step size
            ps = np.sqrt(1 - c_sigma) * ps + np.sqrt(c_sigma * mu_eff) * C_inv_sqrt @ (x_new - mean)
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Update covariance matrix
            diff = x_new - mean
            h_sigma = (1 - c_sigma) ** 2 - c_sigma * (1 - (1 - c_sigma) ** (2 * (calls_made // pop_size)))
            h_sigma = min(h_sigma, 1)
            # Simplified: just use the standard update
            delta = (norm_ps / chi_n - 1) * c_sigma / d_sigma
            pc = (1 - c_c) * pc + h_sigma * c_c * (diff / sigma)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - c_c) * C)
            for i in range(min(mu, pop_size)):
                zi = (Y[i] - mean) / sigma
                C += c_mu * weights[i] * (np.outer(zi, zi) - C)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            C_inv_sqrt = cholesky_inv_sqrt(C)

        return best_x, best_fx
# EVOLVE-BLOCK-END
