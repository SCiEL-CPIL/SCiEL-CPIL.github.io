import numpy as np
import time

class Optimizer:
    """
    Hybrid Differential Evolution + (1+1)-ES for Büche-Rastrigin.
    
    Phase 1: Differential Evolution (DE/rand/1/bin) with adaptive mutation factor
             to explore the multimodal landscape.
    Phase 2: (1+1)-ES with adaptive step size to polish the best solution found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Split budget: 70% for DE, 30% for Polish
        de_budget = int(budget * 0.7)
        polish_budget = budget - de_budget
        
        # --- Phase 1: Differential Evolution ---
        pop_size = 20
        # Ensure pop_size is not too large for the budget
        if de_budget < pop_size:
            pop_size = max(5, de_budget // 2)
            
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        budget_used = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        # DE Parameters
        CR = 0.6
        F_start = 0.8
        F_end = 0.4
        
        # Estimate total generations to plan F decay
        # We don't know exact generations yet, so we use budget_used ratio
        # Or just decay F based on time/iterations count.
        
        # We will run DE until budget_used >= de_budget or time is up
        while budget_used < de_budget:
            if time.time() > deadline - 5.0:  # Leave 5s for polish
                break
            
            # Calculate current F based on progress
            progress = budget_used / de_budget
            F = F_start + (F_end - F_start) * progress
            
            # Generate trial vectors
            for i in range(pop_size):
                if budget_used >= de_budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Select 3 distinct random indices different from i
                idx = np.random.choice(pop_size, 3, replace=False)
                # Ensure none are i (np.random.choice with replace=False might include i if pop_size is small? No, it picks from 0..N-1. If i is in idx, we need to re-pick or handle.)
                # Simpler: pick 3 random, if any is i, replace it.
                while len(set(idx)) < 3 or i in idx:
                    idx = np.random.choice(pop_size, 3, replace=False)
                    # If pop_size is very small, this might loop. But pop_size >= 5.
                
                # DE/rand/1/bin
                # v = x_r1 + F * (x_r2 - x_r3)
                v = population[idx[0]] + F * (population[idx[1]] - population[idx[2]])
                
                # Binomial Crossover
                j_rand = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                budget_used += 1
                
                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Check time again after a full generation
            if time.time() > deadline - 5.0:
                break

        # --- Phase 2: (1+1)-ES Polish ---
        # Start from the best individual found by DE
        x = best_x.copy()
        fx = best_f
        
        # Adaptive sigma
        sigma = 0.5
        min_sigma = 1e-4
        max_sigma = 5.0
        success_window = []
        window_size = 10
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
            else:
                success_window.append(0)
            
            # Adapt sigma
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                if len(success_window) > window_size:
                    success_window.pop(0)
                
                sigma = max(min_sigma, min(sigma, max_sigma))

        return x, fx
