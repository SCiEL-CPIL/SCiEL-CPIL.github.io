import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (self.NP, self.dim))
        f_vals = np.array([func(x) for x in pop])
        calls = self.NP
        
        best_idx = np.argmin(f_vals)
        best_x = pop[best_idx].copy()
        best_f = f_vals[best_idx]
        
        stagnation_counter = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            for i in range(self.NP):
                if calls >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(self.NP, 3, replace=False)
                
                # Mutation
                v = pop[r1] + self.F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                trial = pop[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial < f_vals[i]:
                    pop[i] = trial
                    f_vals[i] = f_trial
                    stagnation_counter = 0
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    stagnation_counter += 1
                    
            # Optional: Adapt F if stagnant
            if stagnation_counter > 20:
                # Increase F to escape local minima
                self.F = min(1.0, self.F * 1.2)
                stagnation_counter = 0

        return best_x, best_f
