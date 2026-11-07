"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 8  # lambda
        mu = pop_size // 2  # number of parents
        weights = np.ones(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.8) + c_sigma) / (
            (dim + 1.3) ** 2 + mu_eff
        )
        c_mu = min(
            1 - c_1,
            (2 * (sqrt_mu_eff - 1.5 + 1 / dim)) /
            ((dim + 2) ** 2 + sqrt_mu_eff),
        ) if (sqrt_mu_eff := np.sqrt(mu_eff)) else 1 - c_1

        # Initialize
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1

        # Precompute
        sqrt_mu_eff = np.sqrt(mu_eff)
        chi_n = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_i = np.clip(x_mean + sigma * (B @ (D * z)), -5, 5)
                pop.append(x_i)

            # Evaluate population
            pop_f = np.array([func(x) for x in pop])
            calls += pop_size

            # Update best
            idx_best = np.argmin(pop_f)
            if pop_f[idx_best] < best_f:
                best_f = pop_f[idx_best]
                best_x = pop[idx_best].copy()

            # Sort by fitness
            order = np.argsort(pop_f)
            x_parents = np.array([pop[i] for i in order[:mu]])
            f_parents = pop_f[order[:mu]]

            # Weighted mean
            x_new = (weights[:, None] * x_parents).sum(axis=0)

            # Update evolution paths
            inv_sqrt_D = 1.0 / D
            # C = B @ diag(D^2) @ B.T
            # For efficiency, work in transformed space
            # Transform x_new - x_mean to the B,D space
            # x_new - x_mean = sigma * B @ D @ z_mean where z_mean is the transformed mean
            # We need to compute the mean in the standard normal space
            # z_mean = D^-1 @ B^T @ (x_new - x_mean) / sigma
            diff = x_new - x_mean
            z_mean = inv_sqrt_D * (B.T @ diff) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            ps_norm = np.linalg.norm(ps)

            if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
                C_update = c_1 * (np.outer(pc, pc) - np.eye(dim)) + c_mu * sum(
                    w * np.outer(z_i, z_i) for w, z_i in zip(weights, inv_sqrt_D * (B.T @ (x_parents - x_mean) / sigma))
                )
            else:
                C_update = c_1 * (np.outer(pc, pc) - np.eye(dim) + c_sigma * (2 - c_sigma) * np.outer(ps, ps) / (1 - (1 - c_c) ** (2 * (calls + 1)))) + c_mu * sum(
                    w * np.outer(z_i, z_i) for w, z_i in zip(weights, inv_sqrt_D * (B.T @ (x_parents - x_mean) / sigma))
                )

            # Update B and D via eigendecomposition
            C = B @ (D ** 2)[:, None] * B.T
            C += C_update
            # Ensure symmetry
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50)

            x_mean = x_new

            # If we've used most of the budget, do a final polish
            remaining = budget - calls
            if remaining <= pop_size:
                # Polish phase: smaller steps around best
                sigma_polish = max(sigma * 0.1, 1e-4)
                while calls < budget:
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma_polish *= 1.1
                    else:
                        sigma_polish *= 0.85
                    if sigma_polish < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
