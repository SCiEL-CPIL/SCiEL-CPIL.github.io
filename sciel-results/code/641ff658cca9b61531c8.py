"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid Differential Evolution and (1+1)-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution for global exploration
        # We allocate 60% of the budget to DE.
        de_budget = int(self.budget * 0.6)
        pop_size = 20
        # Ensure pop_size is not larger than de_budget
        if pop_size > de_budget:
            pop_size = max(2, de_budget)
            
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        fitnesses = np.array([func(ind) for ind in population])
        
        # DE Parameters
        F = 0.8
        CR = 0.9
        
        # Run DE for remaining DE budget
        calls_made = pop_size
        while calls_made < de_budget:
            # Determine how many individuals we can process in this generation
            # We process one individual at a time to strictly control budget
            # But DE usually processes the whole population. 
            # To strictly adhere to budget, we will iterate through individuals 
            # and stop if we exceed budget.
            
            # For simplicity and strict budget control, we do a "semi-online" DE
            # or just run full generations if they fit.
            # Since pop_size=20, and we have ~300 calls, we can do ~15 generations.
            # Let's just do full generations if they fit.
            
            if calls_made + pop_size > de_budget:
                # If a full generation doesn't fit, we stop DE phase
                break
                
            # Update population
            new_fitnesses = fitnesses.copy()
            for i in range(pop_size):
                # Select two distinct random indices
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # Create trial vector
                v = population[i] + F * (population[r1] - population[r2])
                v = np.clip(v, -5, 5)
                
                fv = func(v)
                calls_made += 1
                
                # Selection
                if fv < fitnesses[i]:
                    population[i] = v
                    new_fitnesses[i] = fv
            
            fitnesses = new_fitnesses

        # Find the best individual from DE phase
        best_idx = np.argmin(fitnesses)
        x_best = population[best_idx].copy()
        f_best = fitnesses[best_idx]
        
        # Phase 2: (1+1)-ES for local refinement
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0:
            sigma = 0.5 # Start with a moderate step size
            x = x_best
            fx = f_best
            
            for _ in range(remaining_budget):
                # Generate mutant
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                    
            x_best, f_best = x, fx

        return x_best, f_best
# EVOLVE-BLOCK-END
