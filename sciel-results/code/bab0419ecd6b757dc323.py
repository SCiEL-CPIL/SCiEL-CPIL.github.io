"""Population-based evolutionary strategy for Sharp Ridge (f13) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based evolutionary strategy for Sharp Ridge (f13).
    Maintains a population of candidate points with independent Gaussian perturbations
    and dynamically shrinking sigma to exploit the ridge's anisotropy.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def count_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            xc = np.clip(x, -5, 5)
            f_val = func(xc)
            if f_val < best_f:
                best_f = f_val
                best_x = xc.copy()
            return f_val

        # Population-based evolutionary strategy
        pop_size = 50
        sigma = 2.0  # Initial step size
        sigma_shrink = 0.95  # Multiplicative shrinkage per generation
        sigma_min = 1e-6
        sigma_max = 5.0

        # Initialize population with random points
        population = []
        for i in range(pop_size):
            x = rng.uniform(-4, 4, n)
            fx = count_func(x)
            if fx >= 1e10:
                break
            population.append((fx, x.copy()))

        if not population:
            return best_x, best_f

        # Sort by fitness
        population.sort(key=lambda item: item[0])

        # Track best
        best_fx, best_x_local = population[0]

        # Generations
        max_generations = 100
        elite_count = 5  # Keep top 5 elites each generation

        for gen in range(max_generations):
            if calls >= budget or not time_ok():
                break

            # Generate offspring: for each non-elite, perturb a random parent
            offspring = []
            # Keep elites
            elites = population[:elite_count]
            offspring.extend(elites)

            # Generate new candidates
            for i in range(pop_size - elite_count):
                if calls >= budget or not time_ok():
                    break

                # Pick a random parent (weighted toward better solutions)
                # Use tournament selection
                idx1 = rng.integers(0, min(len(population), 5))
                idx2 = rng.integers(0, min(len(population), 5))
                parent_idx = idx1 if population[idx1][0] < population[idx2][0] else idx2
                parent_x = population[parent_idx][1]

                # Gaussian perturbation
                noise = rng.standard_normal(n) * sigma
                child_x = parent_x + noise
                child_x = np.clip(child_x, -5, 5)

                fy = count_func(child_x)
                if fy >= 1e10:
                    break
                offspring.append((fy, child_x.copy()))

            # Combine population and offspring, sort, keep top pop_size
            population = offspring
            population.sort(key=lambda item: item[0])
            population = population[:pop_size]

            # Update best
            if population[0][0] < best_fx:
                best_fx = population[0][0]
                best_x_local = population[0][1].copy()

            # Adapt sigma: shrink if making progress, grow slightly if stuck
            # Simple adaptation: shrink each generation
            sigma = max(sigma * sigma_shrink, sigma_min)

            # Occasionally try to grow sigma if stuck (no improvement in last few gens)
            # We can track this with a stagnation counter
            # For simplicity, just keep shrinking

        # Final polish: use remaining budget for local refinement around best point
        if calls < budget and time_ok():
            # Do a few more evaluations around the best point with small sigma
            polish_sigma = max(sigma, 1e-4)
            for _ in range(20):
                if calls >= budget or not time_ok():
                    break
                noise = rng.standard_normal(n) * polish_sigma
                candidate = best_x_local + noise
                candidate = np.clip(candidate, -5, 5)
                fy = count_func(candidate)
                if fy >= 1e10:
                    break
                if fy < best_fx:
                    best_fx = fy
                    best_x_local = candidate.copy()
                polish_sigma *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
