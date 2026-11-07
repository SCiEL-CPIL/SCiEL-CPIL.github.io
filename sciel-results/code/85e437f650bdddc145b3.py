"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        pop_size = 50
        F = 0.8
        CR = 0.9

        rng = np.random.default_rng(42)

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        calls_made = pop_size

        # Track progress for early termination
        no_improve_calls = 0
        stagnation_threshold = 150

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Check stagnation
            if no_improve_calls >= stagnation_threshold:
                # Restart with new random individuals to escape local minima
                n_restarts = min(10, budget - calls_made)
                for _ in range(n_restarts):
                    if calls_made >= budget:
                        break
                    new_x = rng.uniform(-5, 5, dim)
                    new_f = func(new_x)
                    calls_made += 1
                    if new_f < best_f:
                        best_x = new_x.copy()
                        best_f = new_f
                        no_improve_calls = 0
                        # Update population
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = new_x
                        fitness[worst_idx] = new_f
                if no_improve_calls >= stagnation_threshold:
                    break

            # DE: generate trial vector for each individual
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select two distinct random indices different from i
                while True:
                    r1 = rng.integers(0, pop_size)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(0, pop_size)
                    if r2 != i and r2 != r1:
                        break

                # Mutation
                v = pop[i] + F * (pop[r1] - pop[r2])
                v = np.clip(v, -5, 5)

                # Crossover (binomial)
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) <= CR
                cross_mask[j_rand] = True  # Ensure at least one gene from v
                trial = np.where(cross_mask, v, pop[i])
                trial = np.clip(trial, -5, 5)

                # Evaluate
                trial_f = func(trial)
                calls_made += 1

                # Selection
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        no_improve_calls = 0
                    else:
                        no_improve_calls += 1
                else:
                    no_improve_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
