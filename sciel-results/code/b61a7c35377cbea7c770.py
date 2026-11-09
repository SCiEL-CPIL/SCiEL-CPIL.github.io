"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution based optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        while calls_made + pop_size <= budget:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                u = np.copy(pop[i])
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Bounds check (optional, func clips anyway, but good practice)
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= new_fitness[i]:
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
            
            pop = new_pop
            fitness = new_fitness
            
        # Use remaining budget for local refinement around best_x
        # Simple random perturbation
        while calls_made < budget:
            # Perturb best_x
            perturbation = rng.normal(0, 0.1, dim)
            candidate = np.clip(best_x + perturbation, -5, 5)
            f_cand = func(candidate)
            calls_made += 1
            if f_cand < best_f:
                best_f = f_cand
                best_x = candidate.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
