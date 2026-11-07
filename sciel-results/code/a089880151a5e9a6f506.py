"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for Weierstrass f16 (multimodal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size: balance exploration vs exploitation
        pop_size = min(40, max(10, 2 * dim + 10))
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.7  # crossover rate

        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_made = pop_size
        gen = 0

        while calls_made < budget:
            if time.time() > deadline:
                break
            gen += 1

            # Adaptive F: decrease slightly over generations
            F_gen = max(0.5, F - 0.3 * gen / 100)
            CR_gen = min(0.9, CR + 0.1 * gen / 100)

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct individuals other than i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential mutation
                v = pop[a] + F_gen * (pop[b] - pop[c])
                # Clip to bounds
                v = np.clip(v, -5, 5)

                # Binary crossover
                trial = pop[i].copy()
                r = rng.random(dim)
                jrand = rng.integers(dim)
                mask = r <= CR_gen
                mask[jrand] = True
                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                ft = func(trial)
                calls_made += 1

                if ft < fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
