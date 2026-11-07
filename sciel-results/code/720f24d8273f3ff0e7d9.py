import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        lambda_ = 20
        F = 0.5
        CR = 0.9
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (lambda_, n))
        fit = np.array([func(x) for x in pop])
        calls_made = lambda_
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # DE Loop
        while calls_made + lambda_ <= budget and time.time() < deadline:
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(lambda_):
                # Select 3 distinct random indices
                indices = [j for j in range(lambda_) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                j_rand = rng.integers(n)
                u = np.empty(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                        
                # Clip
                u = np.clip(u, -5, 5)
                
                # Evaluation
                fu = func(u)
                calls_made += 1
                
                if fu <= new_fit[i]:
                    new_pop[i] = u
                    new_fit[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
            pop = new_pop
            fit = new_fit
            
        # Local Polish
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            sigma = 0.5
            for _ in range(remaining):
                if time.time() > deadline: break
                y = np.clip(best_x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    
        return best_x, best_f
