"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for global optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(42)

        # Initial population
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # Standard CMA-ES pop size
        x0 = np.zeros(dim)
        sigma = 2.0  # Initial step size

        # CMA-ES parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Initialize population
        population = [x0 + sigma * C_sqrt @ rng.standard_normal(dim) for _ in range(pop_size)]
        fitness = [func(np.clip(x, -5, 5)) for x in population]

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_fx = fitness[best_idx]

        # Main loop
        calls_made = pop_size
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            indices = np.argsort(fitness)
            population = [population[i] for i in indices]
            fitness = [fitness[i] for i in indices]

            # Mean update
            mean = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (mean - x0) / sigma)
            p_sigma_norm = np.linalg.norm(p_sigma)

            # Update p_c
            h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - pop_size + 1))) < (1 + 2.0 / dim) * np.sqrt(mu_eff) else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - x0) / sigma

            # Update C
            delta = h_sigma * (1 - c_1 - c_mu) * p_c * np.outer(p_c, np.eye(dim))
            C = (1 - c_1 - c_mu + delta) * C
            for i in range(mu):
                z = (population[i] - x0) / sigma
                C += c_mu * weights[i] * np.outer(z, z)

            # Update x0 and sigma
            x0 = mean
            sigma *= np.exp(c_sigma / 5.0 * (p_sigma_norm / np.sqrt(mu_eff) - 1))

            # Update C_sqrt (use Cholesky decomposition)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            # Generate new population
            population = [x0 + sigma * C_sqrt @ rng.standard_normal(dim) for _ in range(pop_size)]
            fitness = [func(np.clip(x, -5, 5)) for x in population]
            calls_made += pop_size

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fx:
                best_fx = fitness[current_best_idx]
                best_x = population[current_best_idx].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
