"""Differential Evolution with adaptive parameters for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """DE-based optimizer for multimodal BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def eval_point(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            f_val = float(func(x))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Space-filling initialization ---
        # Use Halton sequence for good initial coverage
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def halton_point(index, dim):
            point = []
            for d in range(dim):
                base = primes[d % len(primes)]
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point.append(r)
            return np.array(point) * 10.0 - 5.0

        # Initial population: 20 points from Halton + center
        pop_size = 20
        population = []
        for i in range(1, pop_size + 1):
            pt = halton_point(i, dim)
            population.append(pt)
        population.append(np.zeros(dim))
        pop_size = len(population)

        for pt in population:
            if eval_count >= budget - 30 or time.time() > deadline:
                break
            eval_point(pt)

        # Sort by fitness
        pop_f = np.array([float(func(np.clip(p, -5, 5))) for p in population]) if False else None
        # Actually we already evaluated, let's track fitness separately
        pop_fitness = []
        for pt in population:
            # We already evaluated these, but we need to store their fitness
            # Let's re-evaluate is wasteful. Instead, let's track during eval.
            pass

        # Let's redo this more carefully.
        # Reset state and do a clean implementation.

        # --- Clean Implementation ---
        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        all_fitness = []

        def eval_point_clean(x):
            nonlocal eval_count, best_x, best_f, all_fitness
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            f_val = float(func(x))
            all_fitness.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Initialization
        pop_size = 20
        population = []
        for i in range(1, pop_size + 1):
            pt = halton_point(i, dim)
            population.append(pt)
        population.append(np.zeros(dim))
        pop_size = len(population)

        pop_fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if eval_count >= budget - 30 or time.time() > deadline:
                break
            pop_fitness[i] = eval_point_clean(population[i])

        # --- Phase 2: Differential Evolution ---
        # Adaptive parameters
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        no_improve = 0
        max_no_improve = 30

        # Track best fitness per generation for stagnation detection
        prev_best_f = best_f

        while eval_count < budget - 10 and time.time() < deadline:
            # Sort population by fitness
            order = np.argsort(pop_fitness)
            population = [population[i] for i in order]
            pop_fitness = pop_fitness[order]

            # Check stagnation
            if best_f < prev_best_f - 1e-10:
                no_improve = 0
                prev_best_f = best_f
            else:
                no_improve += 1

            # Adapt parameters based on stagnation
            if no_improve > max_no_improve:
                # Restart with larger mutation
                F = 1.0
                CR = 0.95
                # Perturb population around best
                for i in range(pop_size):
                    if i == 0:
                        continue
                    population[i] = best_x + rng.normal(0, 2.0, dim)
                    population[i] = np.clip(population[i], -5.0, 5.0)
                    if eval_count < budget - 10 and time.time() < deadline:
                        pop_fitness[i] = eval_point_clean(population[i])
                no_improve = 0
                max_no_improve = 50  # Give it more time before next restart
            elif no_improve > max_no_improve // 2:
                F = 0.9
                CR = 0.95

            # Generate offspring
            new_population = []
            new_fitness = []

            for i in range(pop_size):
                if eval_count >= budget - 10 or time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if len(set([i, r1, r2, r3])) == 4:
                        break

                # Mutation: DE/rand/1
                mutant = population[r1] + F * (population[r2] - population[r3])

                # Crossover: Binomial
                cross_point = rng.integers(0, dim)
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]

                trial = np.clip(trial, -5.0, 5.0)
                trial_f = eval_point_clean(trial)

                # Selection: keep the better of parent and trial
                if trial_f <= pop_fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_f)
                else:
                    new_population.append(population[i])
                    new_fitness.append(pop_fitness[i])

            if len(new_population) < pop_size:
                break

            population = new_population
            pop_fitness = np.array(new_fitness)

            # Gradually reduce F for exploitation
            if no_improve == 0:
                F = max(0.5, F * 0.98)

        # --- Phase 3: Local Refinement with Coordinate Descent ---
        if eval_count < budget - 2 and time.time() < deadline:
            # Simple coordinate descent for fine-tuning
            current_x = best_x.copy()
            current_f = best_f
            step = 0.5

            for _ in range(50):
                if eval_count >= budget - 2 or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        trial_x = current_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, -5.0, 5.0)
                        trial_f = eval_point_clean(trial_x)

                        if trial_f < current_f - 1e-10:
                            current_x = trial_x
                            current_f = trial_f
                            improved = True

                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
