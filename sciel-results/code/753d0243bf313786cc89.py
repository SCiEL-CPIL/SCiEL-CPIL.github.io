"""Adaptive Differential Evolution for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Differential Evolution with restarts for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # DE parameters
        pop_size = min(30, max(10, budget // 20))
        F = 0.8       # Scale factor
        CR = 0.9      # Crossover rate
        max_no_improve = 50

        # Initialize population uniformly in [-4, 4]^dim
        pop = rng.uniform(-4, 4, (pop_size, dim))
        fvals = np.array([call_func(pop[i]) for i in range(pop_size)])

        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        no_improve = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + abs(np.min(fvals))):
                break

            new_pop = np.zeros_like(pop)
            new_fvals = np.zeros(pop_size)

            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential evolution mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                f_trial = call_func(trial)

                # Selection
                if f_trial <= fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]

            # Update population
            pop = new_pop
            fvals = new_fvals

            # Update global best
            if fvals[0] < best_f - 1e-14:
                best_f = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Adaptive restart if stagnated
            if no_improve > max_no_improve:
                # Perturb worst half of the population
                half = pop_size // 2
                for i in range(half, pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    pop[i] = best_x + rng.normal(0, 0.5, dim)
                    pop[i] = np.clip(pop[i], -5.0, 5.0)
                    fvals[i] = call_func(pop[i])
                no_improve = 0
                max_no_improve = max(20, max_no_improve // 2)

        return best_x, best_f
# EVOLVE-BLOCK-END
