import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Reserve some budget for local refinement
        # DE phase: ~400 calls, Local phase: ~100 calls
        de_budget = int(budget * 0.8)
        local_budget = budget - de_budget
        
        # DE Parameters
        pop_size = max(10, 5 * dim) # 25 for dim=5
        # Ensure we don't exceed de_budget
        max_generations = de_budget // pop_size
        if max_generations < 1:
            pop_size = de_budget
            max_generations = 1

        F = 0.5
        CR = 0.9
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        # Uniform distribution in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        
        evals_used = pop_size
        
        # DE Loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if evals_used >= de_budget:
                break
                
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if evals_used >= de_budget:
                    break
                    
                # Mutation
                # Strategy: 50% DE/best/1, 50% DE/rand/1
                if rng.random() < 0.5:
                    # DE/best/1
                    r1, r2 = rng.integers(0, pop_size, 2)
                    while r1 == r2 or r1 == i or r2 == i:
                        r1, r2 = rng.integers(0, pop_size, 2)
                    mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
                else:
                    # DE/rand/1
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    while r1 == r2 or r1 == r3 or r2 == r3 or r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = rng.integers(0, pop_size, 3)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                    
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    
                # Boundary handling: Clip to [-5, 5]
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                trial_f = func(trial)
                evals_used += 1
                
                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        best_idx = i

        # Local Refinement Phase
        # Use Nelder-Mead on the best solution found
        if local_budget > 10 and time.time() < deadline:
            # Limit the number of function evaluations in local optimization
            # Nelder-Mead can be verbose, so we wrap it or use options
            # scipy.optimize.minimize with method='Nelder-Mead'
            # maxiter is not directly the function eval count, but we can check time/budget
            
            # We need a wrapper to count evaluations and stop if budget exceeded
            local_evals = 0
            def local_func(x):
                nonlocal local_evals
                if local_evals >= local_budget:
                    return 1e10 # Penalize to stop
                local_evals += 1
                return func(x)
                
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
