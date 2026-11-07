"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: evaluator kills after 330s. Use 295s for safety.
        start_time = time.time()
        max_time = 295.0

        # CMA-ES parameters
        # For dim=5, lambda = 4*(1+3*ln(5)) ≈ 4*(1+4.83) ≈ 23
        lam = 4 * int(1 + 3 * np.log(dim))
        mu = lam // 2  # number of weights

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        f_opt_estimate = None
        sigma = 1.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Initial population
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        def decompose(C):
            """Compute C^{1/2} and C^{-1/2} using eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = decompose(C)

        while calls < budget:
            # Check time limit
            if time.time() - start_time > max_time:
                break

            # Generate population
            z = rng.standard_normal((lam, dim))
            pop = x + sigma * (z @ C_sqrt.T)
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.array([func(pop[i]) for i in range(lam)])
            calls += lam

            # Sort by fitness
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]
            fvals_sorted = fvals[idx]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Update step size
            y = (x - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ y
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) > (1.4 + 1 / dim) * chi_n:
                sigma /= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + c_c * (x - x_old) / sigma
            delta_heavy = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / dim) * chi_n else 0
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (delta_heavy - 1) * c_c * (2 - c_c) * C) +
                 c_mu * np.sum(weights[:, None, None] * np.outer(pop_sorted[:mu] - x_old, pop_sorted[:mu] - x_old), axis=0))

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Recompute square root every 10 iterations for efficiency
            if calls % 10 == 0 or calls < 10:
                C_sqrt, C_sqrt_inv = decompose(C)

            # Early stopping if sigma is very small or no progress
            if sigma < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
