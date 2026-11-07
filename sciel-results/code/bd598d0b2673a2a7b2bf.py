"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        # Population size: typically 5*n to 10*n. With budget 500, let's use 10.
        # This allows ~50 generations.
        pop_size = 10
        if self.budget < pop_size * 2:
            pop_size = max(2, self.budget // 2)

        # Initialize population randomly
        population = rng.uniform(bounds_low, bounds_high, size=(pop_size, n))
        fvals = np.array([func(x) for x in population])
        calls = pop_size

        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]

        # DE Parameters
        F = 0.8  # Mutation scale
        CR = 0.5 # Crossover rate

        while calls < self.budget:
            new_population = []
            new_fvals = []

            # Generate offspring
            for i in range(pop_size):
                if calls >= self.budget:
                    break
                
                # Mutation: DE/rand/1
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == r2:
                    r2 = rng.choice(pop_size)
                
                v = population[r1] + F * (population[r2] - population[i])
                
                # Crossover: Binary
                j_rand = rng.integers(0, n)
                trial = population[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Boundary handling: Clip
                trial = np.clip(trial, bounds_low, bounds_high)

                # Evaluate
                f_trial = func(trial)
                calls += 1
                new_population.append(trial)
                new_fvals.append(f_trial)

                # Selection
                if f_trial < fvals[i]:
                    fvals[i] = f_trial
                    population[i] = trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # If we broke early due to budget, we stop
            if calls >= self.budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
