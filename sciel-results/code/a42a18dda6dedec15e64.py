"""Optimizer for BBOB-style f22 using Differential Evolution with adaptive parameters."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with adaptive F and CR for 5D multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE parameters
        pop_size = max(20, 5 * dim)  # population size
        F_min, F_max = 0.1, 1.0
        CR_min, CR_max = 0.1, 0.9

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        f_pop = np.zeros(pop_size)
        for i in range(pop_size):
            f_pop[i] = func(pop[i])
            calls_made += 1
            if f_pop[i] < best_f:
                best_f = f_pop[i]
                best_x = pop[i].copy()

        # Adaptive parameters
        F = 0.5
        CR = 0.7
        success_count = 0
        total_trials = 0

        while calls_made < budget:
            # Sort population by fitness
            idx = np.argsort(f_pop)
            sorted_pop = pop[idx]
            sorted_f = f_pop[idx]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_pop[0].copy()

            # Adaptive F and CR based on recent success rate
            if total_trials > 10:
                success_rate = success_count / total_trials
                # Increase F when success rate is low, decrease when high
                F = np.clip(0.5 + 0.5 * (0.5 - success_rate), F_min, F_max)
                CR = np.clip(0.7 + 0.2 * (0.5 - success_rate), CR_min, CR_max)

            # Generate trial population
            new_pop = sorted_pop.copy()
            new_f = sorted_f.copy()
            success_this_gen = 0

            for i in range(pop_size):
                if calls_made >= budget:
                    break

                # DE/rand/1/bin mutation
                # Choose three distinct random indices different from i
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)

                # Mutation
                v = sorted_pop[r1] + F * (sorted_pop[r2] - sorted_pop[i])

                # Crossover
                j_rand = rng.integers(0, dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = sorted_pop[i][j]

                # Boundary handling: reflect
                u = np.clip(u, -5, 5)

                # Evaluate trial
                f_trial = func(u)
                calls_made += 1
                total_trials += 1

                # Selection
                if f_trial <= sorted_f[i]:
                    new_pop[i] = u
                    new_f[i] = f_trial
                    success_this_gen += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = u.copy()

            success_count = success_this_gen
            total_trials += pop_size

            # Update population
            pop = new_pop
            f_pop = new_f

            # If no improvement in last several generations, perturb
            # (simple stagnation detection)
            if success_this_gen == 0 and calls_made < budget:
                # Perturb the best individual
                best_idx = np.argmin(f_pop)
                perturbation = rng.normal(0, 0.5, dim)
                perturbed = np.clip(pop[best_idx] + perturbation, -5, 5)
                f_perturbed = func(perturbed)
                calls_made += 1
                if f_perturbed < f_pop[best_idx]:
                    pop[best_idx] = perturbed
                    f_pop[best_idx] = f_perturbed
                    if f_perturbed < best_f:
                        best_f = f_perturbed
                        best_x = perturbed.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
