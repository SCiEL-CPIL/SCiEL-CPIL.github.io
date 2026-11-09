"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Parameters
        pop_size = 10 + 5 * dim  # Standard DE population size
        if pop_size > budget // 2:
            pop_size = max(10, budget // 10)
            
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        rng = np.random.default_rng(42)
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(np.clip(x, -5, 5)) for x in population])
        
        calls = pop_size
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main DE loop
        while calls < budget:
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            # Calculate how many individuals we can evaluate in this generation
            # We need to evaluate offspring for each individual in the population
            # But we must respect the remaining budget
            remaining_budget = budget - calls
            
            # If we can't even evaluate a few, break
            if remaining_budget < 1:
                break
            
            # We will iterate through the population. 
            # To be efficient, we generate one offspring per individual.
            # If remaining_budget < pop_size, we only update the first 'remaining_budget' individuals.
            num_updates = min(pop_size, remaining_budget)
            
            for i in range(num_updates):
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Create trial vector using DE/rand/1/bin
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i
            
            population = new_population
            fitness = new_fitness
            
            # Convergence check: if the best fitness hasn't improved significantly 
            # and the population variance is very low, we might stop.
            # Simple check: if population is identical or nearly identical
            if np.allclose(population, population[0], atol=1e-8):
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
