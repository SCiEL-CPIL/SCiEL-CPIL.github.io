"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based random walk with aggressive pruning for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Initialize population with diverse points
        pop_size = 20
        population = [rng.uniform(-5, 5, d) for _ in range(pop_size)]
        values = [func(x) for x in population]

        best_idx = np.argmin(values)
        best_x = population[best_idx].copy()
        best_f = values[best_idx]

        # Track per-individual sigma for adaptive step size
        sigmas = np.ones(pop_size) * 1.0
        min_sigma = 0.01
        max_sigma = 5.0

        calls_made = pop_size

        while calls_made < budget and time_ok():
            # Sort population by fitness (best first)
            order = np.argsort(values)
            sorted_pop = [population[i] for i in order]
            sorted_vals = [values[i] for i in order]
            sorted_sigmas = [sigmas[i] for i in order]

            # Keep top half, replace bottom half
            keep = pop_size // 2
            new_population = []
            new_values = []
            new_sigmas = []

            # Keep the best individuals
            for i in range(keep):
                new_population.append(sorted_pop[i])
                new_values.append(sorted_vals[i])
                new_sigmas.append(sorted_sigmas[i])

            # Generate new individuals for the rest
            for i in range(keep, pop_size):
                # Pick a random parent from the kept population
                parent_idx = rng.integers(0, keep)
                parent = sorted_pop[parent_idx]
                sigma = sorted_sigmas[parent_idx]

                # Generate offspring
                child = parent + sigma * rng.standard_normal(d)
                child = np.clip(child, -5, 5)

                # Evaluate
                f_child = func(child)
                calls_made += 1

                if not time_ok():
                    break

                new_population.append(child)
                new_values.append(f_child)
                new_sigmas.append(sigma)

                # Update best
                if f_child < best_f:
                    best_x = child.copy()
                    best_f = f_child

                # Adapt sigma based on comparison with parent
                if f_child < sorted_vals[parent_idx]:
                    sigmas[parent_idx] = min(sigma * 1.3, max_sigma)
                else:
                    sigmas[parent_idx] = max(sigma * 0.9, min_sigma)

            if not time_ok():
                break

            # Update population
            population = new_population
            values = new_values
            sigmas = new_sigmas

            # Occasionally re-seed the worst individual with a fresh random point
            if calls_made % 50 == 0:
                worst_idx = np.argmax(values)
                fresh = rng.uniform(-5, 5, d)
                f_fresh = func(fresh)
                calls_made += 1
                if not time_ok():
                    break
                population[worst_idx] = fresh
                values[worst_idx] = f_fresh
                sigmas[worst_idx] = 1.0
                if f_fresh < best_f:
                    best_x = fresh.copy()
                    best_f = f_fresh

        return best_x, best_f
# EVOLVE-BLOCK-END
