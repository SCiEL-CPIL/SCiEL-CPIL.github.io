"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters
        pop_size = 10
        F = 0.8  # Mutation scale factor
        CR = 0.3  # Crossover rate
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        # Spread initial points in [-4, 4] to stay safely inside bounds
        population = rng.uniform(-4, 4, size=(pop_size, n))
        f_values = np.array([func(x) for x in population])
        calls_used = pop_size
        
        global_best_idx = np.argmin(f_values)
        global_best_x = population[global_best_idx].copy()
        global_best_f = f_values[global_best_idx]
        
        # DE Loop
        # Each generation uses pop_size evaluations
        # We have budget - pop_size remaining
        num_generations = (budget - calls_used) // pop_size
        
        for g in range(num_generations):
            # Check if we have enough budget for a full generation
            if budget - calls_used < pop_size:
                break
            
            for i in range(pop_size):
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binomial
                # Choose a random position for crossover
                j_rand = rng.integers(0, n)
                trial = np.copy(population[i])
                
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1
                
                # Update global best
                if f_trial < global_best_f:
                    global_best_f = f_trial
                    global_best_x = trial.copy()
                
                # Selection
                if f_trial <= f_values[i]:
                    population[i] = trial
                    f_values[i] = f_trial
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
