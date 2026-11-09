"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        calls_made = 0

        def count_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Population size
        NP = max(10, 4 * dim)  # 20 for dim=5

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fitness = np.array([count_call(pop[i]) for i in range(NP)])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        # Adaptive parameters
        gen = 0
        no_improve = 0

        while calls_made < budget:
            gen += 1
            prev_best_f = best_f

            # Sort population by fitness
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]

            # Adaptive F and CR based on progress
            if no_improve > 5:
                F = np.clip(F * 0.9, 0.3, 1.0)
                CR = np.clip(CR * 0.95, 0.5, 1.0)

            # Generate offspring
            new_fitness = fitness.copy()
            new_pop = pop.copy()

            for i in range(NP):
                if calls_made >= budget:
                    break

                # DE/rand/1/bin mutation
                # Choose three distinct random individuals different from i
                while True:
                    a = rng.integers(0, NP)
                    b = rng.integers(0, NP)
                    c = rng.integers(0, NP)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break

                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                r = rng.random()
                jrand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                ft = count_call(trial)
                if ft <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

            pop = new_pop
            fitness = new_fitness

            # Check for improvement
            if best_f < prev_best_f - 1e-10:
                no_improve = 0
            else:
                no_improve += 1

            # Restart if stuck
            if no_improve > 15:
                # Keep best, reinitialize rest
                pop[0] = best_x
                fitness[0] = best_f
                for i in range(1, NP):
                    if calls_made >= budget:
                        break
                    pop[i] = rng.uniform(-5, 5, dim)
                    fitness[i] = count_call(pop[i])
                    if fitness[i] < best_f:
                        best_f = fitness[i]
                        best_x = pop[i].copy()
                no_improve = 0
                F = 0.8
                CR = 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
