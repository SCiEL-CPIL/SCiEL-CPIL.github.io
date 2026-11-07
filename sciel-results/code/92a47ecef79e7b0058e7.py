"""Black-box continuous optimizer for BBOB-style Rastrigin (f3)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin

        rng = np.random.default_rng(42)
        total_evals = 0

        # DE parameters
        pop_size = 30  # Must be >= 5 for DE/rand/1
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(x) for x in pop])
        total_evals += pop_size

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # DE loop
        while total_evals < budget and time.time() < deadline:
            # Find best index again in case of updates
            best_idx = np.argmin(fit)
            best_x = pop[best_idx].copy()
            best_f = fit[best_idx]

            # Generate new population
            for i in range(pop_size):
                if total_evals >= budget or time.time() >= deadline:
                    break

                # Select 3 distinct random individuals different from i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                mask[best_idx] = False  # Avoid using best itself in differential (optional, but standard)
                # Actually, standard DE/rand/1 picks 3 distinct from population, can include i? No, usually distinct from i.
                # Let's pick 3 distinct indices from [0, pop_size) excluding i.
                candidates = np.where(mask)[0]
                if len(candidates) < 3:
                    # Fallback if pop_size is too small, but 30 is fine
                    candidates = np.setdiff1d(np.arange(pop_size), [i])
                
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Differential variation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Bound handling: clip to [-5, 5]
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                trial_f = func(trial)
                total_evals += 1
                
                # Selection
                if trial_f < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f

        return best_x, best_f

# EVOLVE-BLOCK-END
