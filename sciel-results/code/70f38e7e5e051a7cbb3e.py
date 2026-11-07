"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive step and local refinement for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize population of 10 individuals
        pop_size = 10
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(population[i]) for i in range(pop_size)])
        budget_used = pop_size

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_fx = fitness[best_idx]

        # DE parameters
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate

        # Phase 1: Differential Evolution (use ~70% of budget)
        de_budget = int(budget * 0.7)
        max_de_iters = de_budget // pop_size

        for gen in range(max_de_iters):
            if budget_used >= budget:
                break
            if time.time() > deadline:
                break

            # Update best
            idx = np.argmin(fitness)
            if fitness[idx] < best_fx:
                best_fx = fitness[idx]
                best_x = population[idx].copy()

            for i in range(pop_size):
                if budget_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)

                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ft = func(trial)
                budget_used += 1

                # Selection
                if ft <= fitness[i]:
                    population[i] = trial
                    fitness[i] = ft

        # Phase 2: Local refinement with (1+1)-ES on the best point
        # Use remaining budget for fine-tuning
        x = best_x.copy()
        fx = best_fx
        sigma = 0.5

        while budget_used < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.05
                if fx < best_fx:
                    best_x, best_fx = x, fx
            else:
                sigma *= 0.95

            sigma = max(0.001, min(sigma, 3.0))

        return best_x, best_fx
# EVOLVE-BLOCK-END
