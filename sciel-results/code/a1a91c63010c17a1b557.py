"""Optimizer for BBOB-style f17 (Schaffers F7) using Differential Evolution."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution for BBOB-style f17 with hidden shift/rotation/offset."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # DE Parameters
        # Population size: typically 5-10 * dim for good exploration
        pop_size = 10 * dim  # 50 for dim=5
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        count = 0
        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_point(x):
            nonlocal count
            try:
                f = func(x)
                count += 1
                return f
            except Exception:
                return float('inf')

        # Initialize population uniformly in the box
        population = np.random.uniform(-5, 5, size=(pop_size, dim))
        fitnesses = np.zeros(pop_size)

        # Evaluate initial population
        for i in range(pop_size):
            fitnesses[i] = eval_point(population[i])
            if fitnesses[i] < best_f:
                best_f = fitnesses[i]
                best_x = population[i].copy()

        # Main DE loop
        while count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Check if we can evaluate a full generation
            if count + pop_size > budget:
                # If not enough budget for a full generation, break
                # or we could do partial, but breaking is safer for consistency
                break

            # Sort population by fitness
            order = np.argsort(fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]

            # Find the best individual in the current sorted population
            # (It's population[0] after sorting)
            # But we need to track the global best separately for return
            # The global best is already tracked in best_x, best_f

            # Generate offspring for each individual
            new_population = np.zeros_like(population)
            new_fitnesses = np.zeros(pop_size)

            for i in range(pop_size):
                if count + pop_size > budget:
                    break

                # Select three distinct random individuals different from i
                # For small pop_size, ensure we don't pick i
                # Using random permutation of indices excluding i
                idxs = [j for j in range(pop_size) if j != i]
                if len(idxs) < 3:
                    # Fallback for very small pop (shouldn't happen with 50)
                    idxs = np.random.choice(pop_size, size=3, replace=False)
                else:
                    idxs = np.random.choice(idxs, size=3, replace=False)
                    idxs = list(idxs)

                r1, r2, r3 = idxs

                # Differential variation
                v = population[r1] + F * (population[r2] - population[r3])

                # Clip to box
                v = np.clip(v, -5, 5)

                # Binomial crossover
                trial = v.copy()
                jrand = np.random.randint(0, dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]

                # Evaluate trial
                f_trial = eval_point(trial)

                # Selection
                if f_trial <= fitnesses[i]:
                    new_population[i] = trial
                    new_fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_population[i] = population[i]
                    new_fitnesses[i] = fitnesses[i]

            # Update population
            population = new_population
            fitnesses = new_fitnesses

        return best_x, best_f
