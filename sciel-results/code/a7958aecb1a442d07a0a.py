import numpy as np
class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Population size
        NP = 20
        if budget < NP:
            NP = budget
            
        # Initialize population
        pop = np.random.uniform(-5, 5, (NP, dim))
        fits = np.zeros(NP)
        
        calls = 0
        for i in range(NP):
            fits[i] = func(pop[i])
            calls += 1
            
        best_idx = np.argmin(fits)
        best_f = fits[best_idx]
        best_x = pop[best_idx].copy()
        
        # DE Parameters
        F = 0.8 # Mutation factor
        
        # Main Loop
        while calls < budget:
            # Select a random individual to mutate
            # To ensure we try to improve the best, we can bias selection
            # or just iterate through the population.
            # Simple approach: For each individual in pop, try to create a better one.
            # But we need to respect the budget.
            
            # Let's pick a random individual index
            idx = np.random.randint(NP)
            
            # Select two distinct random individuals r1, r2
            r1 = np.random.randint(NP)
            r2 = np.random.randint(NP)
            if r1 == idx or r2 == idx or r1 == r2:
                # Retry logic or just pick again
                # Simple fix:
                r1 = np.random.randint(NP)
                r2 = np.random.randint(NP)
                while r1 == idx or r2 == idx or r1 == r2:
                    r1 = np.random.randint(NP)
                    r2 = np.random.randint(NP)
            
            # Create trial vector
            # DE/rand/1
            trial = pop[idx] + F * (pop[r1] - pop[r2])
            
            # Clip to bounds
            trial = np.clip(trial, -5, 5)
            
            # Evaluate
            f_trial = func(trial)
            calls += 1
            
            # Selection
            if f_trial < fits[idx]:
                pop[idx] = trial
                fits[idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    
            # Check if we hit the limit
            if calls >= budget:
                break

        return best_x, best_f
