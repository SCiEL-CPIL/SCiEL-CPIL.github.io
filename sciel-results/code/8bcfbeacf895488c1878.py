import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Parameters
        pop_size = 20
        F = 0.8
        CR = 0.5
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls_made = pop_size
        remaining = self.budget - calls_made
        
        # DE Loop
        while remaining > 0:
            # Update best index
            best_idx = np.argmin(fitness)
            
            new_pop = np.empty_like(pop)
            new_fitness = np.empty_like(fitness)
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: best1bin
                # Use the current best individual in the population
                mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
                
                # Crossover
                mask = rng.random(self.dim) < CR
                # Ensure at least one dimension is from mutant
                if not np.any(mask):
                    mask[rng.integers(self.dim)] = True
                trial = np.where(mask, mutant, pop[i])
                
                # Ensure within bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                remaining -= 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    
            pop = new_pop
            fitness = new_fitness
            
            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_f:
                best_f = fitness[current_best_idx]
                best_x = pop[current_best_idx].copy()
            
            if remaining <= 0:
                break
                
        return best_x, best_f
