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
        
        # Track best solution found so far
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Phase 1: Global Exploration with Adaptive Differential Evolution (JADE-like)
        # Parameters
        pop_size = 12  # Slightly larger for better diversity in rotation
        max_generations = budget // pop_size  # ~41 generations
        
        # Initialize population
        pop = np.zeros((pop_size, dim))
        fit = np.zeros(pop_size)
        
        # First few points: corners and center for quick initial spread
        # Center
        pop[0] = np.zeros(dim)
        # Random points
        for i in range(1, pop_size):
            pop[i] = np.random.uniform(-5, 5, dim)
        
        # Evaluate initial population
        for i in range(pop_size):
            fit[i] = track_call(pop[i])
            
        # Sort initial population
        idx = np.argsort(fit)
        pop = pop[idx]
        fit = fit[idx]
        
        # JADE-style parameters
        F = 0.5
        CR = 0.5
        # Archive of successful vectors (simplified: just keep best few)
        # We will adapt F and CR based on success rates
        
        F_success = []
        CR_success = []
        
        for gen in range(max_generations):
            if time.time() > deadline - 5.0 or calls_used >= budget:
                break
                
            # Generate offspring
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                    
                # Select base vector: current individual
                base = pop[i]
                
                # Select two distinct random indices different from i
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                
                # Differential evolution variant
                # v = pop[r1] + F * (pop[r2] - base)
                v = pop[r1] + F * (pop[r2] - base)
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = np.copy(base)
                # Random crossover
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                f_trial = track_call(trial)
                
                # Selection
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    F_success.append(F)
                    CR_success.append(CR)
                
                # Adapt F and CR (JADE style)
                if len(F_success) >= 10:
                    # Take mean of last 10 successful F/CR
                    F = np.mean(F_success[-10:])
                    CR = np.mean(CR_success[-10:])
                    F_success = F_success[-5:]
                    CR_success = CR_success[-5:]
                    
            # Sort population after generation
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # If we have a very good solution, we can start refining
            if best_f < 1.0:  # Heuristic: if we are close to optimum
                break

        # Phase 2: Local Refinement with (1+1)-ES
        # Start from the best solution found in DE
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with a moderate step size
            sigma_min = 0.001
            sigma_max = 2.0
            recent = []
            local_calls = 0
            remaining = budget - calls_used
            
            # If we still have budget, refine
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                if calls_used >= budget:
                    break
                
                # 1/5 success rule
                if len(recent) >= 10:
                    sr = sum(recent) / 10.0
                    if sr > 0.33:
                        sigma *= 1.15
                    elif sr < 0.1:
                        sigma *= 0.75
                    else:
                        sigma *= 0.95
                    recent = recent[-10:]

                sigma = np.clip(sigma, sigma_min, sigma_max)
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = track_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent.append(1)
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    recent.append(0)

                if len(recent) > 10:
                    recent = recent[-10:]

        return best_x, best_f
