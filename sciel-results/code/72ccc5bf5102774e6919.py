import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # DE parameters
        pop_size = 12
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx]
        best_f = fitness[best_idx]
        
        # DE Loop
        de_budget = int(self.budget * 0.7)
        while eval_count < de_budget:
            if time.time() > deadline:
                break
            for i in range(pop_size):
                if eval_count >= de_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.empty_like(pop[i])
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                trial_f = func(trial)
                eval_count += 1
                
                # Selection
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial
        
        # Local Refinement
        # Use Nelder-Mead for local search
        # We need to count function calls in the callback
        local_calls = [0]
        
        def local_func(x):
            if time.time() > deadline:
                raise KeyboardInterrupt
            local_calls[0] += 1
            return func(x)
        
        # Limit local calls to remaining budget
        remaining_budget = self.budget - eval_count
        if remaining_budget > 0:
            try:
                res = minimize(
                    local_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': remaining_budget, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except KeyboardInterrupt:
                pass
            except Exception:
                pass

        return best_x, best_f
