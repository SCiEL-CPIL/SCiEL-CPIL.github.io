import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        rng = np.random.default_rng(0)
        
        # Population-based Differential Evolution
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, self.dim))
        fit = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        calls_made = pop_size
        remaining_budget = self.budget - calls_made
        
        gen = 0
        while remaining_budget > 0 and time.time() < deadline:
            # Check deadline more frequently
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                if remaining_budget <= 0:
                    break
                if time.time() > deadline:
                    break
                
                # DE/BEST/1 strategy
                r1, r2 = rng.integers(0, pop_size, 2)
                while r1 == r2 or r1 == i or r2 == i:
                    r1, r2 = rng.integers(0, pop_size, 2)
                    
                v = pop[best_idx] + F * (pop[r1] - pop[r2])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                remaining_budget -= 1
                
                # Selection
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i
        
        return best_x, best_f
