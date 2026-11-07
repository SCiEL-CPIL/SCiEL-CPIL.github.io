"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with deadline guard for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # DE parameters
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = np.random.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        calls_made = 0
        for i in range(pop_size):
            fitness[i] = func(population[i])
            calls_made += 1
            if time.time() > deadline - 5.0:
                break

        if calls_made < pop_size:
            pop_size = calls_made
            population = population[:pop_size]
            fitness = fitness[:pop_size]

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Main DE loop
        remaining = budget - calls_made
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break

            # For each individual in the population
            for i in range(pop_size):
                if remaining <= 0:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, size=3, replace=False)
                
                # Mutation: DE/rand/1
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5, 5)

                # Crossover: binomial
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                trial_f = func(trial)
                calls_made += 1
                remaining = budget - calls_made

                # Selection
                if trial_f < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
