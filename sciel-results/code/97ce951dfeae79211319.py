"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin

        # DE Parameters
        pop_size = 20
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]
        population = np.random.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        
        # Evaluate initial population
        calls = 0
        best_x = None
        best_fx = np.inf

        for i in range(pop_size):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            fx = func(population[i])
            calls += 1
            fvals[i] = fx
            if fx < best_fx:
                best_fx = fx
                best_x = population[i].copy()

        if best_x is None:
             # Fallback if no calls were made (should not happen)
             best_x = np.zeros(dim)
             best_fx = func(best_x)
             calls += 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Update population
            for i in range(pop_size):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                r1, r2, r3 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.randint(pop_size)

                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Clip to bounds
                mutant = np.clip(mutant, -5, 5)

                # Crossover
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                ft = func(trial)
                calls += 1

                # Selection
                if ft <= fvals[i]:
                    population[i] = trial
                    fvals[i] = ft
                    if ft < best_fx:
                        best_fx = ft
                        best_x = trial.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
