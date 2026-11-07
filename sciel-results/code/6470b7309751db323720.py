import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 300.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Parameters for Differential Evolution
        NP = 10
        F = 0.8
        CR = 0.9
        
        # Allocate 80% of budget to DE, 20% to local polish
        de_budget = int(budget * 0.8)
        polish_budget = budget - de_budget
        
        # Ensure minimum budget for DE initialization
        if de_budget < NP:
            de_budget = budget
            polish_budget = 0

        rng = np.random.default_rng()
        
        # --- Phase 1: Differential Evolution ---
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        calls_made = NP
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Determine number of generations
        de_calls_remaining = de_budget - NP
        generations = de_calls_remaining // NP
        
        for g in range(generations):
            if time.time() - start_time > timeout:
                break
                
            for i in range(NP):
                # Select two distinct random indices r1, r2 != i
                r1 = rng.integers(0, NP)
                while r1 == i:
                    r1 = rng.integers(0, NP)
                r2 = rng.integers(0, NP)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, NP)

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                mask = rng.random(dim) < CR
                mask[j_rand] = True  # Ensure at least one gene is from mutant
                trial[mask] = v[mask]
                trial[~mask] = pop[i][~mask]
                
                # Boundary handling
                trial = np.clip(trial, -5, 5)
                
                # Evaluation
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i
                        
        # --- Phase 2: (1+1)-ES Polish ---
        
        if polish_budget > 0 and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() - start_time > timeout:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
        return best_x, best_f
