"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE) with a static population for global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Population size: use a moderate size to balance exploration and exploitation
        # With budget=500, pop_size=25 gives 20 generations (500/25)
        # pop_size=50 gives 10 generations
        # Let's use pop_size=25 for more generations but still maintain diversity
        pop_size = 25
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        calls_used = pop_size
        remaining = budget - calls_used

        # DE loop
        while remaining > 0:
            # Determine how many new individuals we can evaluate in this generation
            # Each generation evaluates pop_size new trial individuals
            # But we need to be careful not to exceed budget
            n_to_eval = min(pop_size, remaining)

            # Shuffle indices for crossover reference
            idxs = rng.permutation(pop_size)

            for i in range(n_to_eval):
                # Select three distinct individuals different from i
                while True:
                    a = idxs[i % pop_size]
                    b = idxs[(i + 1) % pop_size]
                    c = idxs[(i + 2) % pop_size]
                    if len({a, b, c, i}) == 4:
                        break

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1
                remaining -= 1

                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            # If we've used up all budget, break
            if remaining <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
