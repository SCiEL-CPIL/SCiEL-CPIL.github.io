"""Rewritten optimizer: population-based DE/rand/1/bin with adaptive parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) with adaptive F and CR."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        NP = 20

        # Initialize population uniformly in [-5, 5]^dim
        pop = np.random.uniform(-5, 5, size=(NP, dim), random_state=rng)
        fvals = np.array([func(x) for x in pop])
        calls_used = NP

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Adaptive parameters
        F = 0.8
        CR = 0.9

        # Track best improvement for adaptive F
        best_improvement = 0.0
        last_best = best_f

        # Main loop
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Update best tracking
            if best_f < last_best:
                best_improvement = last_best - best_f
                last_best = best_f

            # Adaptive F: increase when improving, decrease when stagnating
            if best_improvement > 1e-10:
                F = min(0.95, F + 0.05)
            else:
                F = max(0.5, F - 0.05)

            # Generate offspring for each individual
            for i in range(NP):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random individuals
                idxs = rng.choice(NP, size=4, replace=False)
                a, b, c, j = idxs

                # Mutation: v = a + F * (b - c)
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover: binomial with probability CR
                r = rng.random()
                mask = rng.random(dim) < CR
                mask[j] = True  # Ensure at least one dimension is from v

                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ftrial = func(trial)
                calls_used += 1

                # Selection
                if ftrial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = ftrial
                    if ftrial < best_f:
                        best_f = ftrial
                        best_x = trial.copy()
                        best_idx = i

                # Small local perturbation around best every few iterations
                if calls_used % 50 == 0 and calls_used < budget - 5:
                    # Local search around best
                    local_x = best_x + 0.1 * rng.standard_normal(dim)
                    local_x = np.clip(local_x, -5, 5)
                    flocal = func(local_x)
                    calls_used += 1
                    if flocal < best_f:
                        best_f = flocal
                        best_x = local_x.copy()
                        # Update population member
                        pop[best_idx] = local_x
                        fvals[best_idx] = flocal

        return best_x, best_f
# EVOLVE-BLOCK-END
