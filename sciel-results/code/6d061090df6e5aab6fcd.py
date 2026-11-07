"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant for BBOB f9 (rotated Rosenbrock) in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        popsize = max(4, int(4 + 3 * np.log(n)))  # 9 for n=5
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1) ** 2 + 2 * mu_eff / n) / (
            (n + 2) ** 2 + mu_eff
        )
        c_mu = (
            2 * (mu_eff / (n + 2) ** 2 - 2)
            / ((n + 2 + 2 * mu_eff) ** 2 + mu_eff)
        )
        c_mu = min(c_mu, 2 * (1 - c_1) / (n + 1))

        # Initialization near origin (aligns with basin of attraction)
        x = np.zeros(n)
        sigma = 2.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        inv_C_sqrt = np.eye(n)

        def mat_sqrt_inv(C):
            """Compute sqrt(C) and inv(sqrt(C)) via eigendecomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_vals = np.sqrt(eigvals)
            C_sqrt = eigvecs @ np.diag(sqrt_vals) @ eigvecs.T
            inv_C_sqrt = eigvecs @ np.diag(1.0 / sqrt_vals) @ eigvecs.T
            return C_sqrt, inv_C_sqrt

        C_sqrt, inv_C_sqrt = mat_sqrt_inv(C)

        best_x = x.copy()
        best_f = func(x)
        evals = 1

        # Precompute expected norm of standard normal in n dims
        chi_n = np.sqrt(n * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)))

        while evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            offsets = []
            for _ in range(popsize):
                z = rng.standard_normal(n)
                offsets.append(z)
            offsets = np.array(offsets)

            # Transform and evaluate
            pop = x[:, None] + sigma * (offsets @ inv_C_sqrt.T)
            pop = np.clip(pop, -5, 5)

            fvals = np.empty(popsize)
            for i in range(popsize):
                if evals >= budget:
                    break
                fvals[i] = func(pop[i])
                evals += 1

            # Sort by fitness
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            pop_sorted = pop[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update mean
            y_w = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Update path for sigma adaptation
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * (inv_C_sqrt @ y_w) / sigma

            sigma *= np.exp(
                c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-10, 1e5)

            # Update path for covariance adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * inv_C_sqrt @ y_w

            # Update covariance
            y_w_std = y_w / sigma
            C = (
                (1 - c_1 - c_mu) * C
                + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C)
                + c_mu * np.sum(
                    weights[:, None, None]
                    * np.outer(pop_sorted[:mu, :] - x, pop_sorted[:mu, :] - x)
                    / sigma ** 2,
                    axis=0,
                )
            )

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Recompute sqrt and inv sqrt
            C_sqrt, inv_C_sqrt = mat_sqrt_inv(C)

            # Update mean
            x = np.clip(x + y_w, -5, 5)

            # Stopping criteria
            if sigma < 1e-12:
                break
            if np.all(np.abs(x - best_x) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
