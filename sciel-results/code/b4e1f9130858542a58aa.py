import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        pop_size = 10
        
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fits = np.array([func(x) for x in pop])
        evals = pop_size
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        F = 0.8
        CR = 0.5
        
        start_time = time.time()
        deadline = start_time + 290
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Vectorized Mutation and Crossover for the whole population
            # We need to generate random indices for each individual i
            # r1, r2, r3 for i in 0..pop_size-1
            
            r1 = rng.integers(0, pop_size, pop_size)
            r2 = rng.integers(0, pop_size, pop_size)
            r3 = rng.integers(0, pop_size, pop_size)
            
            # Ensure distinctness is tricky to vectorize perfectly without loops or complex masks.
            # For pop_size=10, the probability of collision is low, but we must handle it.
            # If r1==r2, the mutation vector is 0. 
            # Simple approach: Just accept that some mutations might be weak if indices collide.
            # Or, do it in a loop for correctness. With pop_size=10, a loop is fast enough.
            
            # Let's stick to the loop for correctness and simplicity, as 50 generations * 10 individuals = 500 loops.
            # Python loop overhead is negligible compared to function calls.
            
            for i in range(pop_size):
                # Pick distinct r1, r2, r3 != i
                # Using set to ensure distinctness is easy
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                mask = rng.random(dim) < CR
                mask[j_rand] = True
                
                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                evals += 1
                
                if f_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
