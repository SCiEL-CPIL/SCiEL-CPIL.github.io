import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Differential Evolution
        pop_size = 8
        max_iter = (budget - pop_size) // pop_size
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        calls_made = pop_size
        
        for gen in range(max_iter):
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                v = pop[a] + 0.5 * (pop[b] - pop[c])
                
                # Crossover
                u = v.copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < 0.5 or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, -5, 5)
                fu = func(u)
                calls_made += 1
                
                if fu < fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        return best_x, best_f
