"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Differential Evolution with adaptive step size for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s wall-clock

        rng = np.random.default_rng(42)

        # Initial population: spread across the box
        pop_size = 20
        pop = np.random.uniform(-5, 5, (pop_size, dim), generator=rng)
        fvals = np.array([func(x) for x in pop])
        best_idx = int(np.argmin(fvals))
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls = pop_size

        # Adaptive parameters
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        # Track progress for adaptive F
        prev_best = best_f
        no_improve_count = 0

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Adaptive F: increase exploration if stuck
            if best_f < prev_best - 1e-6:
                no_improve_count = 0
                F = max(0.4, F * 0.9)  # decrease F when improving (exploit)
            else:
                no_improve_count += 1
                if no_improve_count % 10 == 0:
                    F = min(1.5, F * 1.2)  # increase F when stuck (explore)
            prev_best = best_f

            # Generate offspring for each member
            for i in range(pop_size):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct individuals other than i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential mutation
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binary crossover
                mask = rng.random(dim) < CR
                if not mask.any():
                    mask[rng.integers(dim)] = True  # ensure at least one gene from v
                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                ftrial = func(trial)
                calls += 1

                # Selection
                if ftrial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ftrial
                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial.copy()

            # Periodic injection of random individuals to escape local optima
            if no_improve_count > 30 and calls < budget - pop_size:
                # Replace worst few with random points
                worst_indices = np.argsort(fvals)[-3:]
                for idx in worst_indices:
                    if calls >= budget:
                        break
                    new_x = np.random.uniform(-5, 5, dim, generator=rng)
                    fnew = func(new_x)
                    calls += 1
                    pop[idx] = new_x
                    fvals[idx] = fnew
                    if fnew < best_f:
                        best_f = fnew
                        best_x = new_x.copy()
                no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
