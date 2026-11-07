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
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Differential Evolution (Global Search)
        # DE is a population-based stochastic optimizer, distinct from local search
        pop_size = 20
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initial population
        if budget < pop_size:
             # Fallback to simple random search if budget is tiny
             x = rng.uniform(-5, 5, dim)
             f = func(x)
             return x, f

        pop = rng.uniform(-5, 5, (pop_size, dim))
        f_pop = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        # DE Loop
        while calls_made + pop_size <= budget:
            new_pop = np.zeros_like(pop)
            new_f_pop = np.zeros_like(f_pop)
            
            for i in range(pop_size):
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                        
                # Clip to box
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= f_pop[i]:
                    new_pop[i] = u
                    new_f_pop[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                else:
                    new_pop[i] = pop[i]
                    new_f_pop[i] = f_pop[i]
                    
            pop = new_pop
            f_pop = new_f_pop
            
        # Phase 2: Local Polish (Nelder-Mead)
        # Use remaining budget for local refinement to achieve high precision
        remaining = budget - calls_made
        if remaining > 10 and best_x is not None:
            class CallCounter:
                def __init__(self, func, budget_limit):
                    self.func = func
                    self.limit = budget_limit
                    self.count = 0
                    self.exceeded = False
                
                def __call__(self, x):
                    if self.count >= self.limit:
                        self.exceeded = True
                        return 1e10
                    self.count += 1
                    return self.func(x)
            
            counter = CallCounter(func, remaining)
            try:
                res = minimize(counter, best_x, method='Nelder-Mead', 
                               options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
