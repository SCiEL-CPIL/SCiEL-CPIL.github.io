"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(pop_size, n))
        fitness = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls_made = pop_size
        
        # Main loop
        # Each generation uses pop_size evaluations
        while calls_made < budget:
            # If we can't fit a full generation, we might do partial, 
            # but standard DE needs the whole pop for mutation.
            # We have 500 - 10 = 490 left. 490 / 10 = 49 generations.
            # So we can run exactly 49 full generations.
            
            # For each individual in the population
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select 3 distinct random individuals different from i
                # We need 3 indices from {0..pop_size-1} \ {i}
                # pop_size is 10, so we have 9 others.
                candidates = np.setdiff1d(np.arange(pop_size), [i])
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                # Random integer in [0, n-1] to ensure at least one gene from v
                j_rand = rng.integers(n)
                u = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Boundary handling: Clip
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= fitness[i]:
                    pop[i] = u
                    fitness[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
