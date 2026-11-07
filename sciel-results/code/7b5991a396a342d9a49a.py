import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Parameters for Differential Evolution
        pop_size = 10
        CR = 0.8
        F = 0.5
        
        # Initialize population
        # Use a deterministic seed for reproducibility in testing, but random is fine
        # The task says "hidden seeded shift", implying the function is fixed per instance.
        # We should use a fixed seed for our RNG to be consistent? 
        # "For each instance... evaluator creates Optimizer... and calls it once."
        # It doesn't specify if we should use a fixed seed. 
        # Usually, for deterministic evaluation, fixed seed is preferred.
        rng = np.random.default_rng(42)
        
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        call_count = pop_size
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        # Main loop
        while call_count < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            for i in range(pop_size):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct individuals
                # Exclude i from the pool for mutation vectors
                candidates = [j for j in range(pop_size) if j != i]
                a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False)
                
                a = population[a_idx]
                b = population[b_idx]
                c = population[c_idx]
                
                # Mutation
                v = a + F * (b - c)
                
                # Crossover
                u = population[i].copy()
                # Randomly choose a main index to ensure u != v
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                call_count += 1
                
                # Selection
                if f_u < fitnesses[i]:
                    population[i] = u
                    fitnesses[i] = f_u
                    
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
        return best_x, best_f
