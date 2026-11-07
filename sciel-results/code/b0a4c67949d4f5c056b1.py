"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Differential Evolution parameters
        pop_size = 10
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fvals = np.full(pop_size, np.inf)

        # Evaluate initial population
        for i in range(pop_size):
            fx = _eval(population[i])
            if fx is None:
                fvals[i] = np.inf
            else:
                fvals[i] = fx

        # Track best in population
        pop_best_idx = np.argmin(fvals)
        if fvals[pop_best_idx] < best_f:
            best_f = fvals[pop_best_idx]
            best_x = population[pop_best_idx].copy()

        # DE loop
        while calls < budget and time.time() < deadline - 5.0:
            # Sort population by fitness
            order = np.argsort(fvals)
            population = population[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = population[0].copy()

            # Generate offspring
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline - 5.0:
                    break

                # Select three distinct individuals different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Binary crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                fx = _eval(trial)
                if fx is None:
                    break

                # Selection
                if fx <= fvals[i]:
                    population[i] = trial
                    fvals[i] = fx

            # Convergence check: if all individuals are very close
            if np.all(np.abs(population - population[0]) < 1e-8):
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
