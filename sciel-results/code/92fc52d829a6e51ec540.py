"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        calls = 0
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        calls += pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        def check_deadline():
            return calls >= budget or time.time() >= deadline
        
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        
        while not check_deadline():
            # For each individual in population
            for i in range(pop_size):
                if check_deadline():
                    break
                
                # Select 3 distinct random individuals different from i
                while True:
                    r1 = rng.integers(0, pop_size)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(0, pop_size)
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = rng.integers(0, pop_size)
                    if r3 != i and r3 != r1 and r3 != r2:
                        break
                
                # Generate trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.copy(pop[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate
                ft = func(trial)
                calls += 1
                
                # Selection
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
