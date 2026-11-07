"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = min(20, max(8, 2 * dim))
        # Initial population: stratified sampling for better coverage
        pop = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                # Use Latin hypercube-like stratification
                pop[i, d] = -5.0 + 10.0 * (i + rng.random()) / pop_size
        # Ensure within bounds
        pop = np.clip(pop, -5.0, 5.0)

        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls_used = pop_size

        # DE parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Adaptive step: track success rate
        successes = 0
        total_trials = 0

        # Main DE loop
        while calls_used < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]

            # For each individual, create a trial vector
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct individuals different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential mutation
                mutant = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Binary crossover
                trial = sorted_pop[i].copy()
                jrand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = mutant[j]

                # Evaluate trial
                trial_f = func(trial)
                calls_used += 1
                total_trials += 1

                # Selection
                if trial_f <= sorted_fvals[i]:
                    pop[i] = trial
                    fvals[i] = trial_f
                    successes += 1

                    # Update best
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f

            # Adapt F based on success rate in this generation
            gen_success_rate = successes / max(total_trials, 1)
            successes = 0
            total_trials = 0

            # Adjust F: if success rate is high, reduce exploration; if low, increase
            if gen_success_rate > 0.3:
                F = max(0.4, F * 0.9)
            elif gen_success_rate < 0.1:
                F = min(0.95, F * 1.1)

            # Occasionally perturb the worst individual to escape local optima
            if calls_used < budget and rng.random() < 0.1:
                worst_idx = np.argmax(fvals)
                perturbation = rng.normal(0, 1.0, dim)
                new_x = np.clip(pop[worst_idx] + perturbation, -5.0, 5.0)
                new_f = func(new_x)
                calls_used += 1
                if new_f < fvals[worst_idx]:
                    pop[worst_idx] = new_x
                    fvals[worst_idx] = new_f
                    if new_f < best_f:
                        best_x = new_x.copy()
                        best_f = new_f

            # If population has converged, restart with some diversity
            fvar = np.std(fvals)
            if fvar < 1e-6 and calls_used < budget:
                # Restart half the population around the best point
                half = pop_size // 2
                for i in range(half):
                    pop[i] = np.clip(best_x + rng.normal(0, 0.5, dim), -5.0, 5.0)
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_x = pop[i].copy()
                        best_f = fvals[i]

        return best_x, best_f
# EVOLVE-BLOCK-END
