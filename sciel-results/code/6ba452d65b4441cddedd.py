"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for Büche-Rastrigin with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Population size: balance exploration vs. generations
        pop_size = 12
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        calls = pop_size
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        # DE loop
        while calls < budget:
            if time.time() > deadline:
                break
            gen_calls = 0
            for i in range(pop_size):
                if calls + gen_calls >= budget or time.time() > deadline:
                    break
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                trial = np.clip(trial, -5, 5)
                # Evaluate
                ft = func(trial)
                calls += 1
                gen_calls += 1
                # Selection
                if ft < fpop[i]:
                    pop[i] = trial
                    fpop[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
        return best_x, best_f
# EVOLVE-BLOCK-END
