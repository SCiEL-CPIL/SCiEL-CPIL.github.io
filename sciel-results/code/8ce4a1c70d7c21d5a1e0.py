"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Optimizer using Differential Evolution (DE/rand/1/bin).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        NP = 10
        F = 0.8
        CR = 0.5
        
        # Initialize population
        population = np.random.uniform(-5, 5, (NP, dim))
        
        # Evaluate initial population
        fitness = np.zeros(NP)
        calls_made = 0
        best_f = np.inf
        best_x = None
        
        for i in range(NP):
            if calls_made >= budget:
                break
            f = func(population[i])
            fitness[i] = f
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = population[i].copy()
        
        if calls_made >= budget:
            return best_x, best_f
            
        dim_indices = np.arange(dim)
        
        # Main loop
        while calls_made + NP <= budget:
            for i in range(NP):
                if calls_made >= budget:
                    break
                
                # Select three distinct random indices a, b, c != i
                candidates = np.setdiff1d(np.arange(NP), i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                mask = (r <= CR) | (dim_indices == j_rand)
                
                u = np.where(mask, v, population[i])
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= fitness[i]:
                    population[i] = u
                    fitness[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
