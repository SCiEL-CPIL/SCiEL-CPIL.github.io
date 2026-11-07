import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0 # Safety margin
        
        # DE Parameters
        # Population size N. 10 is good for 5D.
        N = 10
        F = 0.8 # Mutation scale
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(N, dim))
        fits = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        evals_used = N
        remaining = budget - N
        
        # Generations
        n_gen = remaining // N
        if n_gen == 0:
            # If budget is too small for even 1 gen, just return best
            # But budget is 500, N=10, so 49 gens.
            pass

        for g in range(n_gen):
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Update population
            for i in range(N):
                if time.time() - start_time > time_limit:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                u = pop[i].copy()
                # Random pivot
                j0 = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j0:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                evals_used += 1
                
                # Selection
                if fu < fits[i]:
                    pop[i] = u
                    fits[i] = fu
                    
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
            # Optional: If no improvement in last few generations, we could restart?
            # With 500 evals, we don't have many restarts.
            # But we can check if we are stuck.
            # For now, standard DE.
            
        return best_x, best_f
