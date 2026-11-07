"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based DE/rand/1/bin for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size: small to maximize generations within budget
        pop_size = 20
        calls_used = 0

        # Initialize population: mix of center and random points for diversity
        population = np.zeros((pop_size, dim))
        # First individual at center
        population[0] = 0.0
        # Remaining: random in [-5, 5]
        for i in range(1, pop_size):
            population[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() > deadline - 5.0 or calls_used >= budget:
                break
            fitness[i] = func(population[i])
            calls_used += 1

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Main DE loop
        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            for i in range(pop_size):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break

                # Mutation: DE/rand/1
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5, 5)

                # Crossover: binomial
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
