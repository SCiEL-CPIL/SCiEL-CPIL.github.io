"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution followed by (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Differential Evolution for global exploration
        # Pop size 10, 35 generations = 350 evaluations
        pop_size = 10
        gen_count = 35
        de_budget = pop_size * gen_count
        
        if budget <= de_budget:
            # Fallback to simple random search if budget is too small for DE
            best_x = np.random.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls = 1
            while calls < budget:
                x = np.random.uniform(-5, 5, dim)
                f = func(x)
                calls += 1
                if f < best_f:
                    best_x, best_f = x, f
            return best_x, best_f

        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # DE Loop
        for g in range(gen_count):
            for i in range(pop_size):
                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.empty_like(pop[i])
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                
                # Selection
                if f_trial < fpop[i]:
                    pop[i] = trial
                    fpop[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        # Phase 2: (1+1)-ES Polish for local refinement
        # Remaining budget
        used = de_budget
        remaining = budget - used
        
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with small step for polishing
            
            for _ in range(remaining):
                y = x + sigma * np.random.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                    
                if sigma < 1e-4:
                    sigma = 1e-4

        return best_x, best_f
# EVOLVE-BLOCK-END
