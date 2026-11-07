"""Differential Evolution with adaptive parameters for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive CR and F."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        N = 20
        # Differential evolution parameters
        F = 0.8
        CR = 0.9

        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        fitness = np.array([func(x) for x in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        calls_made = N

        # Adaptive parameters
        success_count = 0
        total_count = 0

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Adaptive CR based on recent performance
            recent_success_rate = success_count / max(total_count, 1)
            if recent_success_rate < 0.3:
                CR = 0.7
                F = 0.9
            elif recent_success_rate > 0.7:
                CR = 0.95
                F = 0.7
            else:
                CR = 0.9
                F = 0.8

            for i in range(N):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                while r1 == i:
                    r1 = np.random.randint(N)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(N)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.randint(N)

                # Generate trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = np.random.randint(dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True

                trial = np.where(cross_mask, v, pop[i])

                # Evaluate
                ft = func(trial)
                calls_made += 1

                # Selection
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    success_count += 1

                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                else:
                    pass

                total_count += 1

                # Decay success tracking
                if total_count > 50:
                    success_count = success_count * 0.9
                    total_count = total_count * 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
