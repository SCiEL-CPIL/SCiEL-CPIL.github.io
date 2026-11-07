"""Differential Evolution optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pure DE with large static population for global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size: use a good chunk of budget for diversity
        # With 500 calls, we can afford ~50 individuals and multiple generations
        pop_size = 50
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Track calls used
        calls_made = pop_size

        # Main DE loop
        while calls_made < budget:
            # For each individual, create a trial vector
            # We need to be careful about budget: each trial costs 1 call
            # We'll process one individual at a time to respect the budget

            for i in range(pop_size):
                if calls_made >= budget:
                    break

                # Select three distinct random individuals different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover: binomial crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                ft = func(trial)
                calls_made += 1

                # Selection: keep the better
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
