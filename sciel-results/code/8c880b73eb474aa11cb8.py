"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: (1+1)-ES with 1/5 success rule and restarts, plus DE for exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return 1e10
            calls_made += 1
            return func(np.clip(x, -5, 5))

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        # Structural Change: Memetic Algorithm
        # 1. Initialize a small population
        pop_size = 10
        population = []
        fitness = []
        
        for _ in range(pop_size):
            x = rng.uniform(-4, 4, size=dim)
            fx = safe_func(x)
            population.append(x)
            fitness.append(fx)
            try_update(x, fx)

        # 2. Iterative Local Search (Nelder-Mead) on selected points
        # We have ~490 calls left.
        # NM with maxfev=20 uses ~20 calls.
        # We can do ~20 local searches.
        
        while calls_made < budget - 25 and time.time() < deadline:
            # Select a starting point for local search
            # Strategy: 70% Exploit best, 30% Explore random
            min_idx = np.argmin(fitness)
            if rng.random() < 0.7:
                start_idx = min_idx
            else:
                start_idx = rng.integers(0, pop_size)
            
            start_x = population[start_idx].copy()
            start_fx = fitness[start_idx]
            
            # Run Nelder-Mead
            try:
                res = minimize(safe_func, start_x, method='Nelder-Mead', 
                               options={'maxiter': 30, 'maxfev': 25, 'xatol': 1e-6, 'fatol': 1e-6})
                new_x = res.x
                new_fx = res.fun
            except Exception:
                continue
            
            # Update population
            if new_fx < fitness[start_idx]:
                population[start_idx] = new_x
                fitness[start_idx] = new_fx
                try_update(new_x, new_fx)
            else:
                # If no improvement, replace with a new random point to maintain diversity
                # This helps escape local minima
                if rng.random() < 0.5:
                    new_rand = rng.uniform(-4, 4, size=dim)
                    new_rand_fx = safe_func(new_rand)
                    population[start_idx] = new_rand
                    fitness[start_idx] = new_rand_fx
                    try_update(new_rand, new_rand_fx)

        # 3. Final polish on the global best
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                result = minimize(safe_func, global_best_x, method='Nelder-Mead',
                                  options={'maxiter': 50, 'maxfev': budget - calls_made,
                                           'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
