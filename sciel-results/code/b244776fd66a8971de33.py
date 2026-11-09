import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution for Global Search
        NP = 10
        F = 0.8
        CR = 0.9
        
        pop = rng.uniform(-5, 5, (NP, dim))
        fits = np.array([func(x) for x in pop])
        calls = NP
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        stagnant_generations = 0
        max_stagnant = 15
        
        # We will use 80% of the budget for DE
        de_budget = int(budget * 0.8)
        
        while calls < de_budget:
            if time.time() > deadline:
                break
                
            improved_this_gen = False
            for i in range(NP):
                if calls >= de_budget:
                    break
                    
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                trial = np.zeros(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls += 1
                
                if f_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = f_trial
                    improved_this_gen = True
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        stagnant_generations = 0
                else:
                    stagnant_generations += 1
            
            if stagnant_generations > max_stagnant:
                # Restart worst half
                worst_half = NP // 2
                idxs = np.argsort(fits)[worst_half:]
                for idx in idxs:
                    pop[idx] = rng.uniform(-5, 5, dim)
                    fits[idx] = func(pop[idx])
                    calls += 1
                    if fits[idx] < best_f:
                        best_f = fits[idx]
                        best_x = pop[idx].copy()
                stagnant_generations = 0

        # Phase 2: Local (1+1)-ES for Polish
        if calls < budget:
            x_curr = best_x.copy()
            f_curr = best_f
            sigma = 0.5 # Start with a small step
            
            while calls < budget:
                if time.time() > deadline:
                    break
                    
                y = x_curr + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < f_curr - 1e-12:
                    x_curr = y
                    f_curr = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    sigma *= 0.8 # Shrink slowly
                else:
                    sigma *= 1.2 # Grow slowly
                    if sigma > 5.0:
                        # If sigma grows too large, we might have escaped the basin?
                        # Or we are stuck. Let's restart local search from best_x
                        x_curr = best_x.copy()
                        f_curr = best_f
                        sigma = 0.1

        return best_x, best_f
