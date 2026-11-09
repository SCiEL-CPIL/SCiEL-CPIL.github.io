"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive population size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = func(best_x)
        calls_made = 1

        if calls_made >= budget:
            return best_x, best_fx

        # CMA-ES parameters
        lambda_pop = 2 * dim + 3  # population size
        mu = lambda_pop // 2      # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / (dim ** 1.7 + 10)) * (1 - c_c) + 2 * (min(1, (mu_eff - 2 + 1 / mu_eff) / (dim + 2)) - 1) * c_c
        c_mu = (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 1.7 + mu_eff)
        c_mu = min(c_mu, 1 - c_1)

        def cma_es(x0, sigma0, budget_remaining, seed_offset=0):
            nonlocal best_x, best_fx, calls_made

            if budget_remaining <= 0:
                return x0, best_fx

            # Initialize CMA-ES state
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            rng = np.random.default_rng(seed_offset)

            while calls_made < budget:
                # Generate population
                samples = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C @ z)
                    samples.append(y)

                # Evaluate population
                fitness = []
                for i, y in enumerate(samples):
                    if calls_made >= budget:
                        break
                    y_clipped = np.clip(y, -5, 5)
                    fy = func(y_clipped)
                    calls_made += 1
                    fitness.append((fy, i, y_clipped))
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y_clipped.copy()

                if len(fitness) < mu:
                    break

                # Sort by fitness
                fitness.sort(key=lambda x: x[0])
                selected = [f[2] for f in fitness[:mu]]

                # Update mean
                x_new_mean = np.zeros(dim)
                for i, y in enumerate(selected):
                    x_new_mean += weights[i] * y
                x_mean = x_new_mean

                # Update step size (CSA)
                norm_C = np.linalg.norm(C @ np.ones(dim))
                if sigma > 1e-10 and norm_C > 1e-10:
                    z = (x_new_mean - x_mean) / (sigma * norm_C)
                else:
                    z = np.zeros(dim)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (min(100, calls_made) + 1))) / chi_n < 1.4 + 2 / (dim + 1)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update covariance matrix
                delta_sig = (1 - hsig) * c_sigma * (2 - c_sigma)
                pc = (1 - c_c) * pc + delta_sig * (x_new_mean - x_mean) / sigma

                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc + (1 - hsig) * delta_sig * (2 - c_c) * C)
                for i in range(mu):
                    y_i = (selected[i] - x_mean) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Ensure C is positive definite
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                # Check convergence
                if np.linalg.norm(x_new_mean - x_mean) < 1e-10 * max(sigma, 1e-10) or sigma < 1e-8:
                    break

            return x_mean, best_fx

        # Multi-restart CMA-ES
        # First: global exploration with large sigma
        x0 = np.zeros(dim)
        cma_es(x0, 4.0, budget - calls_made, seed_offset=42)

        # If we still have budget, try local refinement from best found
        if calls_made < budget:
            cma_es(best_x.copy(), 0.5, budget - calls_made, seed_offset=123)

        # Final fine-tuning with very small sigma if budget remains
        if calls_made < budget:
            cma_es(best_x.copy(), 0.05, budget - calls_made, seed_offset=456)

        return best_x, best_fx
# EVOLVE-BLOCK-END
