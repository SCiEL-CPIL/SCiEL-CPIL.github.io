"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution based optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # Phase 1: Differential Evolution
        # Population size
        pop_size = 15
        # DE parameters
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Use a mix of random and near-origin to help with Katsuura's global min
        population = np.zeros((pop_size, dim))
        # 1 point at origin
        population[0] = 0.0
        # 7 points near origin
        for i in range(1, 8):
            population[i] = rng.normal(0, 0.5, dim)
        # Rest random in [-5, 5]
        for i in range(8, pop_size):
            population[i] = rng.uniform(-5, 5, dim)
        
        # Evaluate initial population
        pop_fitness = np.zeros(pop_size)
        for i in range(pop_size):
            pop_fitness[i] = wrapped_func(population[i])
        
        # Track best in population
        current_best_idx = np.argmin(pop_fitness)
        
        # DE Loop
        # We have 500 calls. Used 15. Remaining 485.
        # Each generation uses pop_size calls.
        # Max generations = 485 // 15 = 32
        max_generations = 30
        
        for gen in range(max_generations):
            if calls >= self.budget - 10: # Leave budget for polish
                break
            if time.time() - start_time > time_limit - 10.0:
                break
                
            for i in range(pop_size):
                if calls >= self.budget - 10:
                    break
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Ensure they are not i? DE standard allows i, but distinct is better
                # If r1==i, the mutation is x_i + F(x_r2 - x_r3), which is fine.
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5, 5)
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluate trial
                trial_f = wrapped_func(trial)
                
                # Selection
                if trial_f <= pop_fitness[i]:
                    population[i] = trial
                    pop_fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        # Phase 2: Local Polish with Nelder-Mead
        if best_x is not None and calls < self.budget - 5:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: Final small random search if budget remains
        while calls < self.budget - 2:
            if time.time() - start_time > time_limit - 2.0:
                break
            new_p = best_x + rng.normal(0, 0.01, dim)
            new_p = np.clip(new_p, -5, 5)
            v = wrapped_func(new_p)
            if v < best_f:
                best_f = v
                best_x = new_p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
