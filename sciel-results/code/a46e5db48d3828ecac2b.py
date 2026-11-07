"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: (1+1)-ES with 1/5 success rule and restarts, plus DE for exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        # Phase 1: Population-based DE/rand/1/bin with adaptive parameters
        pop_size = 20
        CR = 0.8
        F = 0.8

        # Initialize population: mix of random and local perturbations around a center
        center = rng.uniform(-4, 4, size=dim)
        population = np.zeros((pop_size, dim))
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            if calls_made >= budget:
                break
            if i == 0:
                population[i] = center
            elif rng.random() < 0.5:
                population[i] = center + 1.5 * rng.standard_normal(dim)
            else:
                population[i] = rng.uniform(-4, 4, size=dim)
            population[i] = np.clip(population[i], -5, 5)
            fitness[i] = func(population[i])
            calls_made += 1
            try_update(population[i], fitness[i])

        if calls_made >= budget or global_best_x is None:
            return global_best_x, global_best_fx

        # DE/rand/1/bin main loop
        no_improve_count = 0
        max_no_improve = 40
        current_F = F
        current_CR = CR
        stagnation_rounds = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check for stagnation
            if no_improve_count >= max_no_improve:
                # Diversify: replace worst half with random
                half = pop_size // 2
                order = np.argsort(fitness)
                for i in range(half, pop_size):
                    if calls_made >= budget:
                        break
                    if rng.random() < 0.5:
                        population[i] = global_best_x + 2.0 * rng.standard_normal(dim)
                    else:
                        population[i] = rng.uniform(-4, 4, size=dim)
                    population[i] = np.clip(population[i], -5, 5)
                    fitness[i] = func(population[i])
                    calls_made += 1
                    try_update(population[i], fitness[i])

                no_improve_count = 0
                current_F = F
                current_CR = CR
                stagnation_rounds += 1
                # Reduce max_no_improve slightly to trigger more frequent restarts
                max_no_improve = max(20, max_no_improve - 5)

            # DE/rand/1/bin iteration
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # Create mutant vector
                v = population[r1] + current_F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Binary crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < current_CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                trial_fx = func(trial)
                calls_made += 1

                # Selection
                if trial_fx <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fx
                    if try_update(trial, trial_fx):
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

                # Adapt F and CR based on progress
                if no_improve_count % 10 == 0 and no_improve_count > 0:
                    current_F = np.clip(current_F + 0.1 * rng.standard_normal(), 0.5, 1.2)
                    current_CR = np.clip(current_CR + 0.05 * rng.standard_normal(), 0.5, 1.0)

        # Phase 2: Final local refinement using Scipy Nelder-Mead
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = budget - calls_made

            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return 1e10
                calls_made += 1
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining_budget, 'maxfev': remaining_budget,
                                           'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
