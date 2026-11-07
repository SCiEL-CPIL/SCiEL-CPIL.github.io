"""Optimizer for BBOB f23 (Katsuura) with multi-start DE/rand/1."""
import numpy as np
import time


class Optimizer:
    """Multi-start differential evolution for the Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size: keep it modest to fit within budget and time
        # With dim=5, a population of ~20-30 is reasonable for 500 evals
        pop_size = min(20, max(5, budget // 10))

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])

        best_idx = int(np.argmin(fvals))
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE/rand/1 parameters
        F = 0.8   # differential weight
        CR = 0.9  # crossover rate

        calls_made = pop_size
        remaining = budget - calls_made

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate one trial vector per generation (to conserve budget)
            # Pick a random target
            target_idx = rng.integers(0, pop_size)
            x_target = pop[target_idx]

            # Pick two distinct random individuals different from target
            others = [i for i in range(pop_size) if i != target_idx]
            i1, i2 = rng.choice(others, size=2, replace=False)
            v = pop[i1] + F * (pop[i2] - pop[i1])
            v = np.clip(v, -5, 5)

            # Binomial crossover
            r = rng.random(dim)
            j_rand = rng.integers(0, dim)
            trial = np.where((r < CR) | (np.arange(dim) == j_rand), v, x_target)
            trial = np.clip(trial, -5, 5)

            f_trial = func(trial)
            calls_made += 1

            if f_trial <= fvals[target_idx]:
                pop[target_idx] = trial
                fvals[target_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # Periodic restart: if no improvement for a while, inject diversity
            # Check if population has converged
            if calls_made % 50 == 0:
                f_range = np.max(fvals) - np.min(fvals)
                if f_range < 1e-6:
                    # Restart a fraction of the population
                    n_restart = max(1, pop_size // 4)
                    restart_indices = rng.choice(pop_size, size=n_restart, replace=False)
                    for idx in restart_indices:
                        if idx == best_idx:
                            continue
                        pop[idx] = rng.uniform(-5, 5, dim)
                        fvals[idx] = func(pop[idx])
                        calls_made += 1
                        if fvals[idx] < best_f:
                            best_f = fvals[idx]
                            best_x = pop[idx].copy()
                            best_idx = idx

        return best_x, best_f
