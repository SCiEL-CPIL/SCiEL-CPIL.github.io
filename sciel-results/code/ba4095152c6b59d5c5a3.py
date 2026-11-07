"""Optimized for BBOB f16 (Weierstrass) with time-aware differential evolution + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based optimizer with DE exploration and CMA-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s
        dim = self.dim
        budget = self.budget

        # Track all evaluations for early progress scoring
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse exploration with Differential Evolution
        # Weierstrass is multimodal; DE's population diversity helps escape local minima
        pop_size = 20
        F = 0.5  # differential weight
        CR = 0.9  # crossover rate

        # Initialize population spread across the box
        rng = np.random.default_rng(42)
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        fitness = np.array([evaluate(x) for x in pop])

        # Sort by fitness
        idx = np.argsort(fitness)
        pop = pop[idx]
        fitness = fitness[idx]

        # DE evolution with adaptive parameters
        gen = 0
        max_de_gens = 40
        no_improve = 0
        F_current = F
        CR_current = CR

        while call_count < budget and gen < max_de_gens:
            if time.time() > deadline:
                break

            gen += 1
            improved = False

            for i in range(pop_size):
                if call_count >= budget or time.time() > deadline:
                    break

                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.choice(pop_size)

                # Differential mutation
                mutant = pop[i] + F_current * (pop[r1] - pop[r2]) + 0.5 * F_current * (pop[r3] - pop[i])

                # Binomial crossover
                j_rand = rng.integers(dim)
                cross_mask = rng.random(dim) < CR_current
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, -5, 5)

                f_trial = evaluate(trial)

                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                # Adapt parameters on stagnation
                if no_improve >= 3:
                    F_current = max(0.2, F_current * 0.8)
                    CR_current = min(0.95, CR_current + 0.05)
                    no_improve = 0

            # Resort
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]

        # Phase 2: Local refinement around the best point using (1+1)-ES
        # This polishes the solution found by DE
        if call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            max_es_steps = budget - call_count
            es_step = 0

            while es_step < max_es_steps:
                if time.time() > deadline or call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                es_step += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                    # If sigma gets too small, try a larger jump (restart-like)
                    if sigma < 0.01:
                        # Try a random perturbation from the global best
                        x = best_x + rng.normal(0, 1.0, dim)
                        x = np.clip(x, -5, 5)
                        fx = evaluate(x)
                        sigma = 1.0

        # Phase 3: Final local search with very small steps
        if call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            final_steps = budget - call_count

            for _ in range(final_steps):
                if time.time() > deadline or call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
