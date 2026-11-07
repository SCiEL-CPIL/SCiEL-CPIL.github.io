"""Optimized black-box continuous optimizer using Differential Evolution."""
import numpy as np
import time


class Optimizer:
    """
    Population-based optimizer using Differential Evolution (DE/rand/1)
    with adaptive mutation factor, LHS initialization, stagnation-triggered
    restarts, and a final local polish phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for diverse initial population
        pop_size = max(10, min(20, budget // 15))
        pop_size = max(pop_size, dim + 1)

        # LHS initialization
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            indices = rng.permutation(pop_size)
            for i, idx in enumerate(indices):
                population[idx, d] = -5.0 + (i + 0.5) * 10.0 / pop_size
            population[:, d] += rng.uniform(-0.5 / pop_size, 0.5 / pop_size, pop_size)

        population = np.clip(population, -5.0, 5.0)

        # Evaluate initial population and track fitness
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if calls_made >= budget or time.time() > deadline:
                break
            fitness[i] = evaluate(population[i])
            if np.isinf(fitness[i]):
                population[i] = best_x if best_x is not None else np.zeros(dim)
                fitness[i] = best_f

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)
            fitness[0] = min(fitness[0], best_f)
            population[0] = best_x

        # Phase 2: DE/rand/1 with adaptive parameters
        f_adaptive = 0.8
        cr = 0.9
        stagnation_counter = 0
        max_stagnation = 25  # generations without improvement before restart
        prev_best_f = best_f
        generation = 0

        while calls_made < budget and time.time() < deadline:
            # Check if we have enough budget for one generation
            if budget - calls_made < pop_size:
                break

            improved_this_gen = False
            gen_best_f = best_f

            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # DE/rand/1: pick 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                mask = (r1 != i) & (r2 != i) & (r3 != i)
                if not mask.all():
                    # Regenerate if any equals i
                    while True:
                        r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                        if r1 != i and r2 != i and r3 != i:
                            break

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + f_adaptive * (population[r2] - population[r3])

                # Binomial crossover
                j_rand = rng.integers(dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() <= cr or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                f_trial = evaluate(trial)

                # Selection: keep the better of trial and current population[i]
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < gen_best_f:
                        gen_best_f = f_trial
                        improved_this_gen = True

            generation += 1

            # Update best
            if improved_this_gen:
                # Find the new best in population
                idx_best = np.argmin(fitness)
                best_f = fitness[idx_best]
                best_x = population[idx_best].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive F: decrease F when stagnating, increase when improving
            if improved_this_gen:
                f_adaptive = min(f_adaptive * 1.05, 0.95)
            else:
                f_adaptive = max(f_adaptive * 0.95, 0.4)

            # Stagnation-triggered restart: perturb population
            if stagnation_counter >= max_stagnation:
                # Perturb all population members around best_x
                for i in range(pop_size):
                    if i == np.argmin(fitness):
                        continue
                    population[i] = np.clip(
                        best_x + rng.normal(0, 2.0, dim), -5.0, 5.0
                    )
                    if calls_made < budget and time.time() < deadline:
                        fitness[i] = evaluate(population[i])
                stagnation_counter = 0
                max_stagnation = int(max_stagnation * 0.8)  # tighten over time
                max_stagnation = max(max_stagnation, 10)

        # Phase 3: Final local polish using coordinate descent on best_x
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.5
            min_step = 1e-5
            polish_calls = 0
            max_polish = min(50, budget - calls_made)

            while polish_calls < max_polish and step > min_step and time.time() < deadline:
                improved = False
                for i in range(dim):
                    if polish_calls >= max_polish or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if polish_calls >= max_polish:
                            break
                        x_new = x.copy()
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        if np.allclose(x_new, x):
                            continue
                        fx_new = evaluate(x_new)
                        polish_calls += 1
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            break
                if improved:
                    step *= 1.2
                else:
                    step *= 0.5

        return best_x, best_f
