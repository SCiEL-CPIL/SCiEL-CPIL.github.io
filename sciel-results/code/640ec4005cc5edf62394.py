"""Rewritten optimizer using CMA-ES for the Rosenbrock function."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the Rosenbrock function on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        mean = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = clip(x)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f, x_clipped

        # Initial evaluation
        f0, x0 = evaluate(mean)
        best_x = x0.copy()
        best_f = f0

        # Main CMA-ES loop
        while calls < budget:
            # Save old state for path updates
            old_mean = mean.copy()
            old_B = B.copy()
            old_D = D.copy()

            # Generate population
            pop = []
            pop_f = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x_i = mean + sigma * (B @ (D * z))
                f_i, x_i_clipped = evaluate(x_i)
                if calls >= budget:
                    break
                pop.append(x_i_clipped)
                pop_f.append(f_i)

            if calls >= budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pop = [pop[i] for i in sorted_idx]
            pop_f = [pop_f[i] for i in sorted_idx]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * pop[i]
            mean = new_mean

            # Update evolution paths
            # pc update
            pc = (1 - c_c) * pc + c_c * ((mean - old_mean) / sigma)
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * (D.T @ (B.T @ (mean - old_mean))))) / sigma

            # Update step size (sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            # C = B D^2 B^T
            # Update B and D
            diff = (mean - old_mean) / sigma
            C_update = c_1 * (np.outer(pc, pc) + (1 - c_c) * np.eye(dim) - np.eye(dim)) + c_mu * sum(
                weights[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean) / sigma ** 2 for i in range(mu)
            )
            # Actually, the standard update is:
            # C <- (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - c_c) * I) + c_mu * sum(w_i * (x_i - old_mean)(x_i - old_mean)^T)
            # Let's implement this properly

            # Get old C
            C_old = (B @ np.diag(D ** 2)) @ B.T

            C_new = (1 - c_1 - c_mu) * C_old + c_1 * (np.outer(pc, pc) + (1 - c_c) * np.eye(dim))
            for i in range(mu):
                diff_i = (pop[i] - old_mean) / sigma
                C_new += c_mu * weights[i] * np.outer(diff_i, diff_i)

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_new)
            # Handle numerical issues
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

        return best_x, best_f
# EVOLVE-BLOCK-END
