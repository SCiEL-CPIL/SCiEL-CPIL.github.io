"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution Optimizer for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Parameters for DE/rand/1/bin
        pop_size = 20  # Moderate population size for dim=5
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover rate
        max_iter = 1000  # Max iterations, controlled by budget/time

        # Initialize population uniformly in [-5, 5]
        rng = np.random.default_rng(42)
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([call_func(ind) for ind in population])

        # Track best
        best_idx = np.argmin(fitness)
        current_best_x = population[best_idx].copy()
        current_best_f = fitness[best_idx]

        # Main DE loop
        for gen in range(max_iter):
            if time.time() > deadline or calls_used >= budget:
                break

            for i in range(pop_size):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Ensure they are distinct from i if possible, but DE usually allows it.
                # For robustness, let's just pick 3 distinct indices from the whole population.
                # Standard DE/rand/1/bin: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate trial
                f_trial = call_func(trial)

                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < current_best_f:
                        current_best_f = f_trial
                        current_best_x = trial.copy()
                        best_f = f_trial
                        best_x = trial.copy()

        # If we have budget left, do a quick polish with Coordinate Descent on the best
        if time.time() < deadline and calls_used < budget:
            x = current_best_x.copy()
            fx = current_best_f
            step = 0.5
            # Simple coordinate descent polish
            for _ in range(50): # Limited polish steps
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(dim):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[i] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        if np.all(x_new == x):
                            continue
                        fx_new = call_func(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                        else:
                            step *= 0.5
                    if not improved:
                        step *= 0.5
                if step < 1e-6:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
