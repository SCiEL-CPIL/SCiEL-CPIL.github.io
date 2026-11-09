import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Split budget: 80% for DE, 20% for local polishing
        de_budget = int(budget * 0.8)
        local_budget = budget - de_budget
        
        # --- Differential Evolution Phase ---
        pop_size = 20
        # Ensure pop_size is at least 4 for DE
        pop_size = max(4, min(pop_size, de_budget))
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(ind) for ind in pop])
        calls_made = pop_size
        
        F = 0.8
        CR = 0.9
        
        # Run DE until de_budget is exhausted
        while calls_made < de_budget:
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select 3 distinct individuals different from i
                idxs = [j for j in range(pop_size) if j != i]
                i1, i2, i3 = rng.choice(idxs, size=3, replace=False)
                
                # Mutation
                mutant = pop[i1] + F * (pop[i2] - pop[i3])
                
                # Crossover
                trial = np.copy(pop[i])
                cross_mask = rng.random(dim) < CR
                # Ensure at least one gene is from mutant
                if not np.any(cross_mask):
                    j = rng.integers(0, dim)
                    cross_mask[j] = True
                trial[cross_mask] = mutant[cross_mask]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
        
        # Find best individual
        best_idx = np.argmin(fit)
        x_best = pop[best_idx]
        f_best = fit[best_idx]
        
        # --- Local Polishing Phase (1+1)-ES ---
        if local_budget > 0:
            x = x_best.copy()
            fx = f_best
            sigma = 0.5 # Start with smaller sigma for local search
            
            for _ in range(local_budget):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2 # Increase step if successful
                else:
                    sigma *= 0.8 # Decrease step if unsuccessful
            
            # Update best if local search found better
            if fx < f_best:
                x_best = x
                f_best = fx

        return x_best, f_best
