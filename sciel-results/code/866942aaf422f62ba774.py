"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution with adaptive parameters for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Differential Evolution parameters
        # For Katsuura (f23), which has many local minima due to its oscillatory nature,
        # DE is a good choice as it maintains population diversity.
        
        # We need to be careful with the budget. Each evaluation in DE counts.
        # DE typically evaluates the whole population each generation.
        # Let's use a smaller population to fit within the budget.
        
        # Initial random point to satisfy the "first call" requirement and get a baseline
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, dim)
        best_x = x0.copy()
        best_f = func(best_x)
        
        # If budget is small, just do a simple random search or (1+1)-ES
        if budget < 50:
            sigma = 2.0
            for _ in range(budget - 1):
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            return best_x, best_f

        # Population size for DE
        # Standard DE uses popsize = 5 * dim, but we need to fit in budget.
        # Let's try popsize = 10 * dim, but cap it.
        popsize = 10 * dim
        # Ensure popsize is odd for DE
        if popsize % 2 == 0:
            popsize += 1
            
        # Estimate max generations
        # Each generation evaluates popsize individuals (roughly, including the trial vectors)
        # Actually, DE evaluates popsize trial vectors per generation.
        # So max_gen = (budget - 1) // popsize
        max_gen = (budget - 1) // popsize
        
        if max_gen < 1:
            # Fall back to (1+1)-ES if we can't even do one generation
            sigma = 2.0
            for _ in range(budget - 1):
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            return best_x, best_f

        # Initialize population
        # One member is our current best, rest are random
        population = np.random.uniform(-5, 5, (popsize, dim))
        population[0] = best_x
        
        # Evaluate initial population
        f_values = np.empty(popsize)
        f_values[0] = best_f
        calls_made = 1
        
        for i in range(1, popsize):
            if calls_made >= budget:
                break
            f_values[i] = func(population[i])
            calls_made += 1
            if f_values[i] < best_f:
                best_f = f_values[i]
                best_x = population[i].copy()

        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Main DE loop
        for gen in range(max_gen):
            if calls_made >= budget:
                break
                
            # Find current best index in population
            best_idx = np.argmin(f_values)
            
            # Generate trial vectors
            for i in range(popsize):
                if calls_made >= budget:
                    break
                    
                # Select two distinct random indices different from i
                r1, r2 = rng.choice(popsize, size=2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(popsize, size=2, replace=False)
                
                # Differential evolution mutation
                # v = x_r1 + F * (x_r2 - x_i)
                v = population[r1] + F * (population[r2] - population[i])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                jrand = rng.randint(dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= f_values[i]:
                    population[i] = trial
                    f_values[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Update best index for next generation
            # (Not strictly necessary if we track best_x/best_f globally,
            # but good for potential adaptive strategies)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
