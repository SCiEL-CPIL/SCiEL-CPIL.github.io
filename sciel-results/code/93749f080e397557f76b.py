"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based differential evolution with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Population size
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fits = np.array([func(x) for x in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        calls_made = pop_size

        # DE/rand/1/bin loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                while True:
                    a = rng.integers(pop_size)
                    if a != i:
                        break
                while True:
                    b = rng.integers(pop_size)
                    if b != i and b != a:
                        break
                while True:
                    c = rng.integers(pop_size)
                    if c != i and c != a and c != b:
                        break

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate
                f_trial = func(trial)
                calls_made += 1

                # Selection
                if f_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
