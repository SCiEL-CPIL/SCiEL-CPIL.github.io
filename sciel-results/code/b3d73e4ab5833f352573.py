"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution + (1+1)-ES polish with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 270.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls_used
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Differential Evolution
        def run_de(max_calls):
            nonlocal calls_used, best_x, best_f
            local_calls = 0

            # Population size
            pop_size = 10
            # Initialize population uniformly in [-5, 5]
            population = rng.uniform(-5, 5, size=(pop_size, dim))
            fitness = np.zeros(pop_size)

            for i in range(pop_size):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break
                f_val = func(population[i])
                calls_used += 1
                local_calls += 1
                fitness[i] = f_val
                update_best(population[i], f_val)

            if calls_used >= budget or time.time() > deadline - 10.0:
                return local_calls

            # DE parameters
            F = 0.8  # scaling factor
            CR = 0.9  # crossover rate

            # Track best fitness for convergence check
            best_f_in_pop = np.inf
            no_improve_count = 0

            while local_calls < max_calls:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break

                # Find index of best individual
                best_idx = np.argmin(fitness)
                current_best_f = fitness[best_idx]

                if current_best_f < best_f_in_pop - 1e-10:
                    best_f_in_pop = current_best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > 50:
                    break

                # Generate offspring for each individual
                for i in range(pop_size):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 10.0:
                        break

                    # Select three distinct random individuals
                    while True:
                        r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                        if len({r1, r2, r3, i}) == 4:
                            break

                    # DE/rand/1/bin
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == rng.choice(dim):
                            trial[j] = population[r1][j] + F * (population[r2][j] - population[r3][j])
                        else:
                            trial[j] = population[i][j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls_used += 1
                    local_calls += 1

                    # Selection
                    if f_trial <= fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        update_best(trial, f_trial)

                if calls_used >= budget or time.time() > deadline - 10.0:
                    break

            return local_calls

        # Run DE with most of the budget
        de_budget = 400
        run_de(de_budget)

        # Phase 2: (1+1)-ES polish from best point found
        if best_x is not None:
            x = best_x.copy()
            fx = best_f

            # Adaptive step size for polish
            sigma = 0.5
            polish_budget = 100

            for _ in range(polish_budget):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    update_best(x, fx)
                else:
                    sigma *= 0.9

            # Final fine-tuning
            sigma = 0.1
            for _ in range(20):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    update_best(x, fx)
                else:
                    sigma *= 0.95

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
