import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: must finish within 300s, evaluator kills at 330s
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size: 20 is a good balance for dim=5 and budget=500
        # 500 / 20 = 25 generations
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        # We need to evaluate all initial points
        if pop_size > budget:
            pop_size = budget
            
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.empty(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if time.time() > deadline:
                break
            fitness[i] = func(population[i])
            
        # If we ran out of budget or time during init, return best
        if pop_size >= budget or time.time() > deadline:
            best_idx = np.argmin(fitness[:pop_size])
            return population[best_idx], fitness[best_idx]

        calls_made = pop_size
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        
        # DE Loop
        # We have budget - calls_made evaluations left
        # Each generation consumes pop_size evaluations
        # So we can run floor((budget - calls_made) / pop_size) generations
        
        num_gens = (budget - calls_made) // pop_size
        
        for g in range(num_gens):
            if time.time() > deadline:
                break
                
            improved = False
            new_fitness = fitness.copy()
            new_population = population.copy()
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                    
                # Select three distinct random indices different from i
                # Efficient selection: pick 3 random from range(pop_size), ensure distinct
                # For small pop_size, simple rejection or permutation is fine
                # Using np.random.choice for simplicity, though slightly slower, dim is small
                # Actually, for speed, let's do manual distinct selection
                r1, r2, r3 = i, i, i
                while r1 == i:
                    r1 = rng.integers(0, pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, pop_size)
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover (Binary)
                # Generate random bits for each dimension
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) <= CR
                cross_mask[j_rand] = True # Ensure at least one gene from v
                
                trial = np.where(cross_mask, v, population[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    improved = True
                    if f_trial < best_val:
                        best_val = f_trial
                        best_idx = i
                        
            population = new_population
            fitness = new_fitness
            
            # Early stopping if no improvement for a long time?
            # With only ~25 gens, maybe not necessary, but good for robustness
            # If no improvement in last 5 generations, break?
            # Let's stick to full budget usage unless time is tight, as DE can sometimes take a few gens to kick in.
            
        # Find the global best in the final population
        final_best_idx = np.argmin(fitness)
        best_x = population[final_best_idx]
        best_f = fitness[final_best_idx]
        
        return best_x, best_f
