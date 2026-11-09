"""Differential evolution optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential evolution with aggressive mutation for multimodal Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Population size for differential evolution
        pop_size = 20
        F = 0.8  # aggressive mutation factor for multimodal landscape
        CR = 0.9  # high crossover rate for diversity

        # Initialize population with diverse random points
        population = []
        for i in range(pop_size):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = eval_point(x)
            population.append((x, fx))
            if call_count >= budget:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            eval_point(x0)

        # Sort by fitness (best first)
        population.sort(key=lambda p: p[1])

        # Differential evolution loop
        while call_count < budget and time.time() < deadline:
            # Check if we should stop (time or budget)
            if call_count >= budget or time.time() > deadline:
                break

            # Generate trial vectors using differential evolution
            for i in range(pop_size):
                if call_count >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential evolution: v = x[a] + F * (x[b] - x[c])
                v = population[a][0] + F * (population[b][0] - population[c][0])

                # Binomial crossover: trial vector u
                u = population[i][0].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]

                # Clip to bounds
                u = np.clip(u, -5.0, 5.0)

                # Evaluate trial vector
                fu = eval_point(u)

                # Selection: keep the better of u and population[i]
                if fu <= population[i][1]:
                    population[i] = (u, fu)

            # Re-sort population after each generation
            population.sort(key=lambda p: p[1])

            # Optional: if stuck, inject some random diversity
            # Check if best hasn't improved in last few evaluations
            if len(population) > 0 and population[0][1] == best_f:
                # If no improvement, add some random perturbation to worst members
                for i in range(max(1, pop_size // 4)):
                    if call_count >= budget or time.time() > deadline:
                        break
                    # Perturb the worst (last) member
                    idx = pop_size - 1 - i
                    x_perturbed = population[idx][0] + rng.normal(0, 2.0, dim)
                    x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                    fx = eval_point(x_perturbed)
                    population[idx] = (x_perturbed, fx)
                population.sort(key=lambda p: p[1])

        # Final local refinement around the best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            fine_tune_budget = min(budget - call_count, 30)
            for _ in range(fine_tune_budget):
                if call_count >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
