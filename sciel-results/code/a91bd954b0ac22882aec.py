"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Differential Evolution with local refinement for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20
        F = 0.8   # scale factor
        CR = 0.7  # crossover rate

        # Initialize population
        pop = np.zeros((pop_size, n))
        fvals = np.zeros(pop_size)

        # Seed with diverse starting points
        pop[0] = np.zeros(n)
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-4, 4, n)

        # Evaluate initial population
        for i in range(pop_size):
            fvals[i] = func(np.clip(pop[i], -5, 5))

        calls_used = pop_size
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE main loop
        gen = 0
        while calls_used < budget and time.time() < deadline - 20:
            # Check for timeout
            if time.time() > deadline - 20:
                break

            # Adaptive parameters: decrease F and CR slightly as we progress
            progress = min(1.0, calls_used / budget)
            F_adapt = F * (1.0 - 0.3 * progress)
            CR_adapt = CR * (0.5 + 0.5 * (1.0 - progress))

            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline - 20:
                    break

                # Select three distinct individuals
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break

                # DE/rand/1/bin
                v = pop[a] + F_adapt * (pop[b] - pop[c])

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                ft = func(trial)
                calls_used += 1

                # Selection
                if ft <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

            gen += 1

            # If stuck, inject diversity
            if gen % 10 == 0:
                # Replace worst individuals with new random ones
                worst_idx = np.argmax(fvals)
                if calls_used < budget and time.time() < deadline - 20:
                    pop[worst_idx] = rng.uniform(-4, 4, n)
                    fvals[worst_idx] = func(np.clip(pop[worst_idx], -5, 5))
                    calls_used += 1
                    if fvals[worst_idx] < best_f:
                        best_f = fvals[worst_idx]
                        best_x = pop[worst_idx].copy()

        # Phase 2: Local refinement around the best solution using a small population
        if calls_used < budget - 50 and time.time() < deadline - 10:
            # Use a focused local search with smaller population around best_x
            local_pop_size = 10
            local_pop = np.zeros((local_pop_size, n))
            local_fvals = np.zeros(local_pop_size)

            local_pop[0] = best_x
            local_fvals[0] = best_f
            for i in range(1, local_pop_size):
                local_pop[i] = np.clip(best_x + 0.3 * rng.standard_normal(n), -5, 5)
                local_fvals[i] = func(local_pop[i])
                calls_used += 1

            local_best_idx = np.argmin(local_fvals)
            local_best_f = local_fvals[local_best_idx]

            F_local = 0.5
            CR_local = 0.5

            for gen_local in range(30):
                if calls_used >= budget or time.time() > deadline - 5:
                    break

                for i in range(local_pop_size):
                    if calls_used >= budget or time.time() > deadline - 5:
                        break

                    while True:
                        a = rng.integers(0, local_pop_size)
                        b = rng.integers(0, local_pop_size)
                        c = rng.integers(0, local_pop_size)
                        if a != i and b != i and c != i and a != b and b != c and a != c:
                            break

                    v = local_pop[a] + F_local * (local_pop[b] - local_pop[c])
                    trial = local_pop[i].copy()
                    j_rand = rng.integers(0, n)
                    for j in range(n):
                        if rng.random() < CR_local or j == j_rand:
                            trial[j] = v[j]

                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls_used += 1

                    if ft <= local_fvals[i]:
                        local_pop[i] = trial
                        local_fvals[i] = ft

                local_best_idx = np.argmin(local_fvals)
                local_best_f = local_fvals[local_best_idx]

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_pop[local_best_idx].copy()

                # Gradually reduce step size
                F_local *= 0.95
                CR_local *= 0.98

        # Phase 3: Final fine-tuning with very small steps
        if calls_used < budget - 10 and time.time() < deadline - 2:
            sigma = 0.01
            x = best_x.copy()
            fx = best_f

            for _ in range(20):
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
