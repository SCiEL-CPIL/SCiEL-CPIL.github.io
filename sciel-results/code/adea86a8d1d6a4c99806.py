"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Differential Evolution optimizer for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = 20

        # Initialize population
        pop = np.zeros((pop_size, dim))
        fvals = np.zeros(pop_size)

        for i in range(pop_size):
            pop[i] = rng.uniform(-5, 5, dim)
            fvals[i] = func(pop[i])

        calls_made = pop_size

        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_fx = fvals[best_idx]

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        mutation = 'DE/rand/1'

        # Adaptive parameters
        no_improve = 0
        max_no_improve = 30

        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Generate offspring for each individual in population
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random individuals different from i
                while True:
                    r1 = rng.integers(pop_size)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(pop_size)
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = rng.integers(pop_size)
                    if r3 != i and r3 != r1 and r3 != r2:
                        break

                # Differential mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binary crossover
                trial = pop[i].copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = mutant[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                ftrial = func(trial)
                calls_made += 1

                # Selection
                if ftrial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = ftrial

                    if ftrial < best_fx:
                        best_fx = ftrial
                        best_x = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            # If stuck for too long, adapt strategy
            if no_improve >= max_no_improve:
                no_improve = 0
                max_no_improve = max(15, max_no_improve - 5)

                # Try to inject diversity: reinitialize worst individuals
                worst_indices = np.argsort(fvals)[-pop_size // 4:]
                for idx in worst_indices:
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    # Local perturbation around best
                    new_point = best_x + rng.standard_normal(dim) * 1.5
                    new_point = np.clip(new_point, -5, 5)
                    fnew = func(new_point)
                    calls_made += 1
                    pop[idx] = new_point
                    fvals[idx] = fnew
                    if fnew < best_fx:
                        best_fx = fnew
                        best_x = new_point.copy()

                # Also try a global restart for one individual
                if calls_made < budget and time.time() < deadline:
                    new_point = rng.uniform(-4, 4, dim)
                    fnew = func(new_point)
                    calls_made += 1
                    worst_idx = np.argmax(fvals)
                    pop[worst_idx] = new_point
                    fvals[worst_idx] = fnew
                    if fnew < best_fx:
                        best_fx = fnew
                        best_x = new_point.copy()

        # Final local refinement with Nelder-Mead if time permits
        if time.time() < deadline - 20 and calls_made < budget:
            try:
                from scipy.optimize import minimize

                remaining = budget - calls_made
                if remaining > 10:
                    def wrapper(z):
                        nonlocal calls_made
                        if calls_made >= budget or time.time() > deadline:
                            return best_fx
                        calls_made += 1
                        return func(np.clip(z, -5, 5))

                    result = minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': max(1, remaining // 4), 'xatol': 1e-8, 'fatol': 1e-10}
                    )

                    if result.fun < best_fx:
                        best_fx = result.fun
                        best_x = np.clip(result.x, -5, 5)

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
