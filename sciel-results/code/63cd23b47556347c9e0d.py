"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Differential Evolution
        # Population size
        pop_size = 10
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, n))
        pop_f = np.array([evaluate(x) for x in pop])

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE loop
        while calls_made < total_budget and time.time() < deadline:
            # Find current best in population
            best_idx = np.argmin(pop_f)
            current_best_x = pop[best_idx].copy()
            current_best_f = pop_f[best_idx]

            # Generate trial vectors
            for i in range(pop_size):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and a != c:
                        break

                # DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = np.copy(pop[i])
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate
                trial_f = evaluate(trial)

                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

            # Update global best
            best_idx = np.argmin(pop_f)
            if pop_f[best_idx] < best_f:
                best_f = pop_f[best_idx]
                best_x = pop[best_idx].copy()

        # Final polishing with local ES
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Fine ES around best
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(30):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                sigma = max(sigma, 1e-6)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
