import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # DE Parameters
        NP = 15  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fitness = np.zeros(NP)
        calls = 0
        
        # Evaluate initial population
        for i in range(NP):
            x = np.clip(pop[i], -5, 5)
            f_val = func(x)
            calls += 1
            pop[i] = x
            fitness[i] = f_val
            
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        while calls + NP <= budget:
            for i in range(NP):
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # Create trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_val = func(trial)
                calls += 1
                
                # Selection
                if f_val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_val
                    if f_val < best_f:
                        best_f = f_val
                        best_x = trial.copy()
                        
        return best_x, best_f
