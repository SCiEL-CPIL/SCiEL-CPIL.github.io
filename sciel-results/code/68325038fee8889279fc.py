"""Adaptive Differential Evolution optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Differential Evolution with restarts for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Population size
        N = 15

        # Initialize population
        pop = np.array([rng.uniform(-4, 4, dim) for _ in range(N)])
        pop_f = np.array([call_func(x) for x in pop])

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        stagnation = 0
        max_stagnation = 50
        gen = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Track best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

            # Generate offspring
            new_pop = [pop[0].copy()]  # elite
            new_f = [pop_f[0]]

            for i in range(1, N):
                if calls >= budget or time.time() > deadline:
                    break

                # Select three distinct individuals different from i
                candidates = [j for j in range(N) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate
                trial_f = call_func(trial)
                new_pop.append(trial)
                new_f.append(trial_f)

            if len(new_pop) < N:
                break

            # Replace population
            pop = np.array(new_pop)
            pop_f = np.array(new_f)

            # Check for stagnation
            if pop_f[0] >= best_f * 1.0001:
                stagnation += 1
            else:
                stagnation = 0

            gen += 1

            # Adaptive restart if stagnated
            if stagnation >= max_stagnation:
                # Keep best, reinitialize rest
                for i in range(1, N):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Mix best with random perturbation
                    x = best_x + rng.standard_normal(dim) * 2.0
                    x = np.clip(x, -5.0, 5.0)
                    pop[i] = x
                    pop_f[i] = call_func(x)

                # Also try some fully random points
                for i in range(N // 2, N):
                    if calls >= budget or time.time() > deadline:
                        break
                    x = rng.uniform(-4, 4, dim)
                    pop[i] = x
                    pop_f[i] = call_func(x)

                stagnation = 0
                max_stagnation = int(max_stagnation * 1.5)
                F = max(0.3, F * 0.8)  # reduce step size

        return best_x, best_f
# EVOLVE-BLOCK-END
