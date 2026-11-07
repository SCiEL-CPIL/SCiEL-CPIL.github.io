import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        dim = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        N = 10  # Population size
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        f_pop = np.array([func(x) for x in pop])
        calls = N
        
        x_best = pop[np.argmin(f_pop)]
        f_best = np.min(f_pop)
        x_ever = x_best.copy()
        f_ever = f_best
        
        # DE Loop
        while calls + N <= budget and time_ok():
            for i in range(N):
                # Select r1, r2, r3 distinct from i
                indices = [j for j in range(N) if j != i]
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i, j]
                        
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = pop[i].copy()
                        if f_best < f_ever:
                            f_ever = f_best
                            x_ever = x_best.copy()
                            
            # Check for stagnation/diversity loss
            spread = np.mean(np.max(pop, axis=0) - np.min(pop, axis=0))
            if spread < 0.2:
                # Re-inject diversity by replacing worst individuals
                worst_indices = np.argsort(f_pop)[-3:]
                for idx in worst_indices:
                    pop[idx] = rng.uniform(-5, 5, dim)
                    f_pop[idx] = func(pop[idx])
                    calls += 1
                    if f_pop[idx] < f_ever:
                        f_ever = f_pop[idx]
                        x_ever = pop[idx].copy()
                        f_best = f_ever
                        x_best = x_ever.copy()
                    
        # Use remaining budget for local search
        sigma = 0.5
        while calls < budget and time_ok():
            n_cand = min(5, budget - calls)
            candidates = x_best + sigma * rng.standard_normal((n_cand, dim))
            candidates = np.clip(candidates, -5, 5)
            f_cands = np.array([func(x) for x in candidates])
            calls += n_cand
            
            best_i = np.argmin(f_cands)
            if f_cands[best_i] < f_best:
                x_best = candidates[best_i]
                f_best = f_cands[best_i]
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()
                sigma *= 1.1
            else:
                sigma *= 0.8
            sigma = np.clip(sigma, 1e-5, 2.0)

        return x_ever, f_ever
