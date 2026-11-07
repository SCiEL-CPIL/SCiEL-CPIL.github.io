import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 300 # Safety margin
        calls = 0
        dim = self.dim
        bounds = (-5.0, 5.0)
        
        # Phase 1: Differential Evolution
        pop_size = 10
        # Initialize population
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        fit = np.array([func(p) for p in pop])
        calls += pop_size
        
        best_idx = np.argmin(fit)
        global_best = pop[best_idx].copy()
        global_best_fit = fit[best_idx]
        
        F = 0.8 # Mutation factor
        CR = 0.9 # Crossover rate
        
        # Run DE for a certain number of generations
        # Budget for DE: ~200 calls
        max_de_calls = 200
        while calls < max_de_calls and time.time() < self.deadline:
            for i in range(pop_size):
                if calls >= max_de_calls or time.time() >= self.deadline:
                    break
                
                # Select random individuals
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                
                # Mutation
                mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip
                trial = np.clip(trial, bounds[0], bounds[1])
                
                # Evaluate
                ft = func(trial)
                calls += 1
                
                # Selection
                if ft < fit[i]:
                    pop[i] = trial
                    fit[i] = ft
                    if ft < global_best_fit:
                        global_best_fit = ft
                        global_best = trial.copy()
                        best_idx = i
            # Update best_idx if it changed? No, best_idx is updated in selection.
            # Actually, best_idx should point to the global best in the population.
            # The code above updates best_idx only if the new individual is the global best.
            # We should ensure best_idx is always the index of the min fit in pop.
            # It's safer to recompute or just track global_best separately.
            # The mutation uses pop[best_idx]. If best_idx is stale, it's bad.
            # Let's update best_idx after each generation or just use global_best for mutation (DE/1 vs DE/rand).
            # Standard DE uses best/rand. Let's stick to using the current global best vector for mutation.
            
        # Phase 2: Local Search (Polishing)
        # Use remaining budget
        x = global_best.copy()
        fx = global_best_fit
        sigma = 0.5
        
        while calls < self.budget and time.time() < self.deadline:
            y = x + sigma * np.random.randn(dim)
            y = np.clip(y, bounds[0], bounds[1])
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2
            else:
                sigma *= 0.7
            if sigma < 1e-6:
                break
                
        return x, fx
