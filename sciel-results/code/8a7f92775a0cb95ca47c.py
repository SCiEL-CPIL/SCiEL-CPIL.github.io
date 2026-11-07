"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + local refinement for Step Ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = (-5.0, 5.0)

        # Phase 1: Differential Evolution for global exploration
        pop_size = 12
        de_budget = int(budget * 0.75)  # 75% for DE
        remaining_budget = budget - de_budget

        # Initialize population
        population = rng.uniform(bounds[0], bounds[1], (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls = pop_size

        best_idx = int(np.argmin(fitness))
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8  # Differential weight - higher for better exploration
        CR = 0.7  # Crossover rate

        stagnation_count = 0
        max_stagnation = 8

        while calls < de_budget:
            # Check time
            if time.time() > deadline:
                break

            # Adapt parameters based on stagnation
            if stagnation_count > max_stagnation:
                # Increase exploration when stuck
                F = min(F + 0.1, 1.0)
                CR = min(CR + 0.1, 1.0)
                # Inject random individual to escape local optima
                if rng.random() < 0.6:
                    idx = int(rng.integers(0, pop_size))
                    population[idx] = rng.uniform(bounds[0], bounds[1], dim)
                    fitness[idx] = func(population[idx])
                    calls += 1
                    if fitness[idx] < best_f:
                        best_f = fitness[idx]
                        best_x = population[idx].copy()
                        best_idx = idx
                stagnation_count = 0

            improved = False
            for i in range(pop_size):
                if calls >= de_budget or time.time() > deadline:
                    break

                # Select 3 distinct random individuals different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, bounds[0], bounds[1])

                # Binomial crossover
                trial = population[i].copy()
                j_rand = int(rng.integers(0, dim))
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate and select
                f_trial = func(trial)
                calls += 1

                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    improved = True

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i

            if not improved:
                stagnation_count += 1
            else:
                stagnation_count = 0

        # Phase 2: Local refinement using coordinate descent + random perturbations
        # This exploits the step nature of f7 - once near the minimum,
        # fine-tuning each coordinate can find the sharp drop
        if calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            local_calls = 0
            max_local_calls = remaining_budget

            # Coordinate descent with adaptive step
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
            for step in step_sizes:
                if local_calls >= max_local_calls or time.time() > deadline:
                    break

                improved_coord = False
                for _ in range(3):  # Multiple passes per step size
                    for j in range(dim):
                        if local_calls >= max_local_calls or time.time() > deadline:
                            break

                        # Try positive direction
                        x_try = x_local.copy()
                        x_try[j] = min(bounds[1], x_local[j] + step)
                        f_try = func(x_try)
                        local_calls += 1
                        calls += 1

                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            improved_coord = True
                        elif local_calls >= max_local_calls or time.time() > deadline:
                            break
                        else:
                            # Try negative direction
                            x_try = x_local.copy()
                            x_try[j] = max(bounds[0], x_local[j] - step)
                            f_try = func(x_try)
                            local_calls += 1
                            calls += 1

                            if f_try < f_local:
                                x_local = x_try
                                f_local = f_try
                                improved_coord = True

                if not improved_coord:
                    # No improvement with this step size, try smaller
                    continue

            # Final random perturbations around best
            if calls < budget and time.time() < deadline:
                for _ in range(min(50, budget - calls)):
                    if time.time() > deadline:
                        break
                    x_try = x_local + 0.01 * rng.standard_normal(dim)
                    x_try = np.clip(x_try, bounds[0], bounds[1])
                    f_try = func(x_try)
                    calls += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
        return best_x, best_f
# EVOLVE-BLOCK-END
