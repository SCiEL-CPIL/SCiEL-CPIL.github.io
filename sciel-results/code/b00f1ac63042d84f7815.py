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
        
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Multi-Start Differential Evolution
        n_restarts = 5
        pop_size = 10
        F = 0.8 # Mutation scale factor
        CR = 0.9 # Crossover rate
        
        # Estimate calls per restart
        # Init: pop_size
        # Generations: (budget_per_restart - pop_size) / pop_size
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if not time_left() or call_count >= budget - 10:
                break
            
            # Initialize population
            P = rng.uniform(-5, 5, (pop_size, dim))
            F_vals = np.array([safe_func(p) for p in P])
            
            for i in range(pop_size):
                if F_vals[i] < best_f:
                    best_f = F_vals[i]
                    best_x = P[i].copy()

            # Run DE generations
            while call_count < budget - 10 and time_left():
                new_P = P.copy()
                new_F = F_vals.copy()
                
                for i in range(pop_size):
                    if call_count >= budget - 5:
                        break
                    
                    # Select distinct indices
                    idxs = rng.choice(pop_size, 3, replace=False)
                    while idxs[0] == i or idxs[1] == i or idxs[2] == i:
                        idxs = rng.choice(pop_size, 3, replace=False)
                    
                    p1, p2, p3 = P[idxs[0]], P[idxs[1]], P[idxs[2]]
                    
                    # Mutation
                    v = p1 + F * (p2 - p3)
                    
                    # Crossover
                    j_rand = rng.integers(dim)
                    u = rng.random(dim) < CR
                    u[j_rand] = True
                    trial = np.where(u, v, P[i])
                    trial = np.clip(trial, -5, 5)
                    
                    f_trial = safe_func(trial)
                    
                    if f_trial <= new_F[i]:
                        new_P[i] = trial
                        new_F[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                P, F_vals = new_P, new_F

        # Phase 2: Local Refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - 1 - call_count
            
            for _ in range(remaining):
                if not time_left() or call_count >= budget - 1:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                sigma = np.clip(sigma, 0.001, 2.0)

        return best_x, best_f
