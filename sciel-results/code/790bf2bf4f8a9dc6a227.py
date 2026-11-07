import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Population size: 30 is a good balance for dim=5
        pop_size = 30
        
        # Reserve budget for local search
        # Let's use 50% for DE and 50% for Local Search initially, 
        # but allow DE to stop early if stagnated.
        de_budget = budget // 2
        local_search_budget = budget - de_budget
        
        # Initialize
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        F = 0.7
        CR = 0.7
        
        generations = 0
        max_generations = (de_budget - calls_made) // pop_size + 10
        stagnation_counter = 0
        stagnation_threshold = 5
        
        while calls_made < de_budget and generations < max_generations:
            gen_best_f = best_f
            
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select r1, r2, r3 distinct from i and each other
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if len(set([r1, r2, r3, i])) == 4:
                        break
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            generations += 1
            
            # Stagnation check
            if best_f == gen_best_f:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            if stagnation_counter >= stagnation_threshold:
                break
        
        # Local Search Phase
        remaining_calls = budget - calls_made
        
        if remaining_calls > 0:
            # Get top 3 candidates from population
            # Sort population by fitness
            sorted_indices = np.argsort(fitnesses)
            top_k = min(3, pop_size)
            top_candidates = [population[idx].copy() for idx in sorted_indices[:top_k]]
            
            # Distribute remaining budget among top candidates
            # Give more to the best one
            budgets = []
            total_weight = 0
            for k in range(top_k):
                w = (top_k - k) # Best gets top_k, next gets top_k-1, etc.
                budgets.append(w)
                total_weight += w
            
            for k, x0 in enumerate(top_candidates):
                if remaining_calls <= 0:
                    break
                alloc = int(remaining_calls * budgets[k] / total_weight)
                if alloc < 10: alloc = 10 # Minimum for NM
                if alloc > remaining_calls: alloc = remaining_calls
                
                try:
                    res = minimize(
                        func, 
                        x0, 
                        method='Nelder-Mead',
                        options={'maxfev': alloc, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                    remaining_calls -= alloc
                except Exception:
                    pass

        return best_x, best_f
