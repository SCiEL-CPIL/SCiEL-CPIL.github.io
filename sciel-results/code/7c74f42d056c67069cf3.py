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
        
        # Track best
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Differential Evolution for exploration
        # Use ~70% of budget for DE
        de_budget = int(budget * 0.7)
        np_size = 10
        max_gen = de_budget // np_size
        
        # Initialize population
        pop = rng.uniform(-5, 5, (np_size, dim))
        f_pop = np.array([func(x) for x in pop])
        
        # Update best
        idx = np.argmin(f_pop)
        best_x = pop[idx].copy()
        best_f = f_pop[idx]
        
        # DE loop
        calls_made = np_size
        F = 0.8
        CR = 0.9
        
        for g in range(max_gen):
            for i in range(np_size):
                if calls_made >= de_budget:
                    break
                
                # Select two distinct random indices
                r1, r2 = rng.choice([j for j in range(np_size) if j != i], size=2, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                trial = pop[i].copy()
                cross_mask = rng.random(dim) < CR
                # Ensure at least one gene is mutated
                if not np.any(cross_mask):
                    j = rng.integers(0, dim)
                    cross_mask[j] = True
                trial[cross_mask] = v[cross_mask]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # Phase 2: Local Polish with Nelder-Mead or similar
        # Use remaining budget
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0:
            # Use scipy's minimize with Nelder-Mead
            # maxiter is the number of function evaluations? 
            # scipy minimize Nelder-Mead maxiter is number of iterations, not evals.
            # We need to be careful with the budget.
            
            # Let's implement a simple local search or use scipy with a wrapper to count calls.
            # Actually, scipy's minimize doesn't easily allow strict eval budget control without a wrapper.
            # Let's write a simple (1+1)-ES polish, which is easier to control.
            
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_budget = remaining_budget
            
            for _ in range(polish_budget):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.85
                    
                if sigma < 1e-6:
                    break
                    
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
