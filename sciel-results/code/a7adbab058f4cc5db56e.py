"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for BBOB f12 (Bent Cigar) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget
        low = -5.0
        high = 5.0

        # Population size
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        rng = np.random.default_rng(42)

        # Initialize population
        population = rng.uniform(low, high, (pop_size, dim))
        fitnesses = np.array([func(ind) for ind in population])
        calls_made = pop_size

        # Track best
        best_idx = int(np.argmin(fitnesses))
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]

        # Differential Evolution loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Pick a random target
            i = int(rng.integers(0, pop_size))

            # Select 3 distinct individuals excluding i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = rng.choice(candidates, size=3, replace=False)

            # Mutation: v = population[a] + F * (population[b] - population[c])
            v = population[a] + F * (population[b] - population[c])

            # Crossover
            trial = np.empty(dim)
            r = int(rng.integers(0, dim))
            for j in range(dim):
                if rng.random() < CR or j == r:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]

            # Clip to bounds
            trial = np.clip(trial, low, high)

            # Evaluate
            f_trial = func(trial)
            calls_made += 1

            # Selection
            if f_trial <= fitnesses[i]:
                population[i] = trial
                fitnesses[i] = f_trial

                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = i

        return best_x, best_f
# EVOLVE-BLOCK-END
