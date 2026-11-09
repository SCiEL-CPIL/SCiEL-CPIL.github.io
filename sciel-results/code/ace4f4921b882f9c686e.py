"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Population-based Differential Evolution optimizer for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # DE parameters
        pop_size = 24
        F = 0.8
        CR = 0.9

        # Initialize population: mix of origin, random, and best_x perturbations
        population = np.zeros((pop_size, n))
        f_values = np.zeros(pop_size)
        
        # First individual is origin
        population[0] = np.zeros(n)
        f_values[0] = best_f
        
        # Second individual is a random point
        population[1] = rng.uniform(-4, 4, n)
        f_values[1] = do_call(population[1])
        if f_values[1] < best_f:
            best_f = f_values[1]
            best_x = population[1].copy()

        # Rest of the population: random points
        for i in range(2, pop_size):
            if check_deadline() or calls_used >= budget:
                break
            population[i] = rng.uniform(-4, 4, n)
            f_values[i] = do_call(population[i])
            if f_values[i] < best_f:
                best_f = f_values[i]
                best_x = population[i].copy()

        # Main DE loop
        while calls_used < budget - 10 and not check_deadline():
            # Find the index of the best individual in the current population
            best_idx = np.argmin(f_values)
            
            # Update global best
            if f_values[best_idx] < best_f:
                best_f = f_values[best_idx]
                best_x = population[best_idx].copy()

            # Generate offspring for each individual
            for i in range(pop_size):
                if calls_used >= budget - 10 or check_deadline():
                    break

                # Select three distinct random indices different from i
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    r3 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # DE/rand/1 mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(0, n)
                trial = np.zeros(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = do_call(trial)
                
                # Selection
                if f_trial <= f_values[i]:
                    population[i] = trial
                    f_values[i] = f_trial

        # Final local polish using Nelder-Mead from the best DE point
        remaining_budget = budget - calls_used
        if remaining_budget > 20 and not check_deadline():
            def objective(x):
                nonlocal calls_used
                if check_deadline() or calls_used >= budget:
                    raise ValueError("budget exceeded")
                val = func(np.clip(x, -5, 5))
                calls_used += 1
                return val

            try:
                res = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_budget - 5, 200),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
