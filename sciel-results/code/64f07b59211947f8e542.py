"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # ~32 for dim=5
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        calls_made = 1
        x_best = x0.copy()
        f_best = f0

        mean = x0.copy()
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # inverse square root of C
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Helper to compute inverse square root of covariance matrix
        def inv_sqrt(C_mat):
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-10)
            inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T

        C_sqrt_inv = inv_sqrt(C)

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate lambda_ individuals
            population = []
            fitnesses = []

            for i in range(lambda_):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                population.append(y)
                fitnesses.append(fy)

            if len(population) == 0:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]

            # Update best
            for i in range(len(population)):
                if fitnesses[i] < f_best:
                    f_best = fitnesses[i]
                    x_best = population[i].copy()

            # Compute new mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i]

            # Update step size control variable ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (new_mean - mean)) / sigma

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update covariance matrix
            delta_h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_))) < (1 + 2 / (dim + 1)) * chi_n else 0
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = C_sqrt_inv @ (population[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c_1 * delta_h * (c_c * (2 - c_c) - 1) - c_1) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            mean = new_mean
            C_sqrt_inv = inv_sqrt(C)

            # Stagnation check: if no improvement in last lambda_ evaluations, restart
            if len(fitnesses) > 0 and fitnesses[0] >= f_best + 1e-10 and calls_made < budget * 0.7:
                # Try a random restart
                x_restart = rng.uniform(-5, 5, dim)
                f_restart = func(x_restart)
                calls_made += 1
                if f_restart < f_best:
                    f_best = f_restart
                    x_best = x_restart.copy()
                # Reset CMA-ES state
                mean = x_best.copy()
                sigma = 2.0
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)

        # Final polish
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline:
            sigma_polish = 0.05
            x = x_best.copy()
            fx = f_best
            for i in range(min(remaining, 30)):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 0.9
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                else:
                    sigma_polish *= 0.95
                if sigma_polish < 1e-8:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
