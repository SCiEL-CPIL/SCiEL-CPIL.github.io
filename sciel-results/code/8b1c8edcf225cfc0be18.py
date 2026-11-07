"""Adaptive Multi-start Differential Evolution for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive DE/rand/1 with population of 40, multi-start diversity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size - smaller for more frequent full sweeps
        pop_size = 40

        # Initialize population with structured diversity
        pop = np.zeros((pop_size, dim))
        # Center point
        pop[0] = 0.0
        # Near-center points
        pop[1] = rng.uniform(-0.5, 0.5, dim)
        pop[2] = rng.uniform(-1, 1, dim)
        # Mid-range points
        pop[3] = rng.uniform(-2, 2, dim)
        pop[4] = rng.uniform(-3, 3, dim)
        # Far points
        pop[5] = rng.uniform(-4, 4, dim)
        # Remaining random
        for i in range(6, pop_size):
            pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_used = pop_size

        # DE parameters - start with good exploration
        F = 0.7
        CR = 0.8

        # Track for adaptive strategies
        no_improve = 0
        gen = 0
        best_f_history = []

        while calls_used < budget:
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Adaptive F and CR based on stagnation and generation
            if no_improve > 30:
                F = 0.5
                CR = 0.9
            elif no_improve > 15:
                F = 0.6
                CR = 0.85
            else:
                # Gradually reduce F as we progress
                progress = min(1.0, calls_used / budget)
                F = 0.7 - 0.2 * progress
                CR = 0.8 + 0.1 * progress

            improved = False

            # Generate offspring for each member
            for i in range(pop_size):
                if calls_used >= budget:
                    break

                # Select three distinct random indices different from i
                # Use tournament selection for better exploitation
                candidates = rng.choice(pop_size, size=3, replace=False)
                a, b, c = candidates

                # Ensure they're different from i
                while a == i or b == i or c == i:
                    candidates = rng.choice(pop_size, size=3, replace=False)
                    a, b, c = candidates

                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover - binomial
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                calls_used += 1

                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    improved = True

                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # If stuck for too long, inject diversity
            if no_improve > 20:
                # Replace worst 25% with new random points
                n_replace = max(1, pop_size // 4)
                for i in range(pop_size - n_replace, pop_size):
                    if calls_used >= budget:
                        break
                    pop[i] = rng.uniform(-5, 5, dim)
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                no_improve = 0

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
