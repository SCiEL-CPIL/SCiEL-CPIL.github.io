"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Differential Evolution optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Stratified + random initial sampling
    - DE/rand/1/bin with adaptive F and CR
    - Tournament-based selection
    - Coordinate descent polish at the end
    - Time guard to stay within wall-clock limits
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling (mix of random and Halton-like quasi-random)
        n_samples = min(50, budget // 10)
        best_x = None
        best_fx = float('inf')
        population = []

        # Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(n_samples):
            if i < n_samples // 2:
                # Halton quasi-random
                x = np.array([halton(i + 1, p) * 10.0 - 5.0 for p in [2, 3, 5, 7, 11][:dim]])
            else:
                # Random
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            population.append((fx, x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Phase 2: Differential Evolution
        pop_size = max(10, min(40, budget // 12))
        F_low, F_high = 0.4, 0.9
        CR = 0.7
        stall_count = 0
        max_stall = 15
        F = 0.7
        best_pop_fx = best_fx

        # Sort initial population
        population.sort(key=lambda t: t[0])

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                base = population[i][1]
                mutant = base + F * (population[r1][1] - population[r2][1]) + F * (population[r3][1] - base)
                mutant = np.clip(mutant, -5, 5)

                # Binomial crossover
                trial = base.copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, -5, 5)

                ft = func(trial)
                calls_used += 1
                offspring.append((ft, trial))

            if len(offspring) < pop_size:
                break

            # Selection: keep better of parent and offspring
            new_population = []
            for i in range(pop_size):
                if offspring[i][0] <= population[i][0]:
                    new_population.append(offspring[i])
                else:
                    new_population.append(population[i])

            population = new_population
            population.sort(key=lambda t: t[0])

            # Update best
            if population[0][0] < best_fx:
                best_fx = population[0][0]
                best_x = population[0][1].copy()
                stall_count = 0
                # Adapt F: decrease F when improving (exploit more)
                F = max(F_low, F * 0.95)
            else:
                stall_count += 1
                # Adapt F: increase F when stagnating (explore more)
                F = min(F_high, F * 1.1)

            # Stagnation restart: inject random individuals
            if stall_count >= max_stall:
                n_inject = max(3, pop_size // 5)
                for _ in range(n_inject):
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_used += 1
                    # Replace worst individual
                    if fx < population[-1][0]:
                        population[-1] = (fx, x)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                population.sort(key=lambda t: t[0])
                stall_count = 0
                F = 0.8  # Reset F to mid-range
                max_stall = min(max_stall + 5, 30)

        # Phase 3: Coordinate descent polish
        if calls_used < budget and time.time() < deadline - 2.0:
            x = best_x.copy()
            fx = best_fx
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            for step in step_sizes:
                if calls_used >= budget or time.time() > deadline - 1.0:
                    break
                improved = True
                while improved and calls_used < budget and time.time() < deadline - 1.0:
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls_used += 1
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                improved = True
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                break  # Found improvement in this direction, try next dimension

        return best_x, best_fx
# EVOLVE-BLOCK-END
