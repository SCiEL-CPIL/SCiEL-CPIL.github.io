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
        
        # Split budget: 70% for DE, 30% for local polishing
        de_budget = int(budget * 0.7)
        local_budget = budget - de_budget
        
        # --- Differential Evolution Phase ---
        # Use larger population for better exploration
        pop_size = min(30, de_budget // 2)
        pop_size = max(8, pop_size)
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(ind) for ind in pop])
        calls_made = pop_size
        
        # Adaptive DE parameters
        F_start, F_end = 0.9, 0.4
        CR_start, CR_end = 0.9, 0.5
        
        total_generations = (de_budget - calls_made) // pop_size
        generation = 0
        
        # Run DE until de_budget is exhausted
        while calls_made < de_budget:
            generation += 1
            progress = min(1.0, generation / max(1, total_generations))
            
            # Adapt F and CR
            F = F_start + (F_end - F_start) * progress
            CR = CR_start + (CR_end - CR_start) * progress
            
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
        x_best = pop[best_idx].copy()
        f_best = fit[best_idx]
        
        # --- Local Polishing Phase (1+1)-ES with adaptive sigma ---
        if local_budget > 0:
            x = x_best.copy()
            fx = f_best
            sigma = 1.0  # Start with moderate sigma
            
            # Track success rate for better sigma adaptation
            successes = 0
            total_steps = 0
            window_size = 10
            
            for step in range(local_budget):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.1  # Increase step if successful
                else:
                    sigma *= 0.9  # Decrease step if unsuccessful
                
                total_steps += 1
                
                # Adapt sigma based on success rate in recent window
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate < 0.15:
                        sigma *= 0.7  # Too many failures, shrink more
                    elif success_rate > 0.45:
                        sigma *= 1.3  # Too many successes, grow more
                    
                    # Reset window
                    successes = 0
            
            # Update best if local search found better
            if fx < f_best:
                x_best = x
                f_best = fx

        return x_best, f_best
