"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for bent cigar optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Population size for DE
        pop_size = 20
        if budget < pop_size + 10:
            pop_size = max(2, budget // 5)
            
        # Initialize population uniformly in the box [-5, 5]
        population = rng.uniform(-5, 5, size=(pop_size, n))
        fitness = np.array([func(x) for x in population])
        budget_used = pop_size
        
        # Track best
        best_idx = np.argmin(fitness)
        x_best = population[best_idx].copy()
        fx_best = fitness[best_idx]
        
        # DE Parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Main loop
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            # If not, we can still do a partial generation or single step
            # For simplicity and robustness, we do full generations if possible
            # If budget < pop_size, we break or do a simple random search
            if budget_used + pop_size > budget:
                # Do a few random perturbations around best if budget allows
                while budget_used < budget and time.time() < deadline:
                    if rng.random() < 0.5:
                        # Local search
                        perturbation = rng.standard_normal(n) * 0.1
                        x_trial = np.clip(x_best + perturbation, -5, 5)
                    else:
                        # Random restart
                        x_trial = rng.uniform(-5, 5, n)
                    
                    f_trial = func(x_trial)
                    budget_used += 1
                    
                    if f_trial < fx_best:
                        x_best = x_trial
                        fx_best = f_trial
                break
            
            # Standard DE Step
            new_population = []
            new_fitness = []
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if budget_used + 1 > budget:
                    break
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Ensure they are distinct from i if possible, but DE allows r's to be any
                # Actually, standard DE requires r1, r2, r3 distinct from each other, 
                # and typically distinct from i for the target, but mutation uses them.
                # Let's just pick 3 random indices.
                
                # Mutation: x_trial = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                x_trial = population[i].copy()
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        x_trial[j] = v[j]
                
                # Clip to bounds
                x_trial = np.clip(x_trial, -5, 5)
                
                # Evaluate
                f_trial = func(x_trial)
                budget_used += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population.append(x_trial)
                    new_fitness.append(f_trial)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            if not new_population:
                break
                
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < fx_best:
                fx_best = fitness[current_best_idx]
                x_best = population[current_best_idx].copy()
            
        return x_best, fx_best
# EVOLVE-BLOCK-END
