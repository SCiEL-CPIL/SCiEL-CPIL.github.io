"""Rewritten program for BBOB f7 (Step ellipsoidal) using a fresh design."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start population-based search with adaptive step for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        x_best = np.clip(rng.uniform(-5, 5, self.dim), -5, 5)
        f_best = func(x_best)
        calls = 1

        # Scout phase: evaluate a spread of points to find a good region
        n_scouts = min(40, self.budget - 1)
        for i in range(n_scouts):
            if calls >= self.budget or time.time() >= deadline:
                break
            # Mix of random points and points near origin
            if i % 3 == 0:
                x = rng.uniform(-2, 2, self.dim)
            elif i % 3 == 1:
                x = rng.uniform(-4, 4, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Main search: population-based with adaptive sigma
        lam = 12  # population size
        sigma = 1.5
        stagnant = 0
        max_stagnant = 15
        pop_x = np.array([x_best + sigma * rng.standard_normal(self.dim) for _ in range(lam)])
        pop_x = np.clip(pop_x, -5, 5)

        while calls < self.budget and time.time() < deadline:
            # Evaluate population
            pop_f = np.empty(lam)
            for i in range(lam):
                if calls >= self.budget or time.time() >= deadline:
                    break
                pop_f[i] = func(pop_x[i])
                calls += 1
            else:
                # All evaluated
                idx = np.argmin(pop_f)
                if pop_f[idx] < f_best:
                    f_best = pop_f[idx]
                    x_best = pop_x[idx].copy()
                    stagnant = 0
                    sigma *= 1.3  # increase step on improvement
                else:
                    stagnant += 1
                    sigma *= 0.75  # decrease step on stagnation

                # Adapt sigma to avoid too small or too large
                sigma = np.clip(sigma, 0.01, 5.0)

                # If stuck too long, do a larger jump (restart-like)
                if stagnant >= max_stagnant:
                    # Try a few larger jumps
                    for _ in range(3):
                        if calls >= self.budget or time.time() >= deadline:
                            break
                        x_try = x_best + 2.0 * rng.standard_normal(self.dim)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < f_best:
                            f_best = f_try
                            x_best = x_try.copy()
                            stagnant = 0
                            sigma = 1.0
                    stagnant = 0
                    sigma = 1.0

                # Generate next population around current best
                pop_x = np.array([x_best + sigma * rng.standard_normal(self.dim) for _ in range(lam)])
                pop_x = np.clip(pop_x, -5, 5)

        return x_best, f_best
# EVOLVE-BLOCK-END
