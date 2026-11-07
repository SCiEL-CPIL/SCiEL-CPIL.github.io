import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        N = 10
        if self.budget < N:
             # Fallback to (1+1)-ES
             x = rng.uniform(-5, 5, self.dim)
             fx = func(x)
             sigma = 2.0
             for _ in range(self.budget - 1):
                 y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                 fy = func(y)
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.5
                 else:
                     sigma *= 0.87
             return x, fx

        pop = rng.uniform(-5, 5, (N, self.dim))
        fpop = np.array([func(x) for x in pop])
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        remaining = self.budget - N
        F = 0.8
        CR = 0.9
        stagnation_count = 0
        prev_best = best_f
        
        while remaining >= N:
            improved = False
            for i in range(N):
                # Select r1, r2, r3
                others = [j for j in range(N) if j != i]
                r1, r2, r3 = rng.choice(others, size=3, replace=False)
                
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        pass
                    else:
                        mutant[j] = pop[i][j]
                
                mutant = np.clip(mutant, -5, 5)
                f_mutant = func(mutant)
                remaining -= 1
                
                if f_mutant < fpop[i]:
                    pop[i] = mutant
                    fpop[i] = f_mutant
                    if f_mutant < best_f:
                        best_f = f_mutant
                        best_x = mutant.copy()
                        improved = True
            
            if improved:
                stagnation_count = 0
            else:
                stagnation_count += 1
                # If stuck, perturb the worst half of the population
                if stagnation_count >= 5:
                    worst_half = np.argsort(fpop)[N//2:]
                    pop[worst_half] = rng.uniform(-5, 5, (len(worst_half), self.dim))
                    fpop[worst_half] = np.array([func(x) for x in pop[worst_half]])
                    remaining -= len(worst_half)
                    # Update best if new randoms are better (unlikely but possible)
                    for i in worst_half:
                        if fpop[i] < best_f:
                            best_f = fpop[i]
                            best_x = pop[i].copy()
                    stagnation_count = 0
        
        # Use remaining budget for local refinement
        if remaining > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining):
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            best_x, best_f = x, fx
            
        return best_x, best_f
