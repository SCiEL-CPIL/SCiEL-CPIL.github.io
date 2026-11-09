import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        # Differential Evolution based optimizer
        pop_size = 15
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate
        init_range = 4.0  # initial population spread
        
        # Initialize population
        population = rng.uniform(-init_range, init_range, size=(pop_size, dim))
        fitness = np.empty(pop_size)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        # Evaluate initial population
        for i in range(pop_size):
            if evals_used >= budget or time.time() > deadline:
                break
            f_val = func(population[i])
            evals_used += 1
            fitness[i] = f_val
            if f_val < best_f:
                best_f = f_val
                best_x = population[i].copy()
        
        # Main DE loop
        generation = 0
        while evals_used < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if budget - evals_used < pop_size:
                break
            
            for i in range(pop_size):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Select three distinct random indices different from i
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and a != c:
                        break
                
                # Create trial vector using DE/rand/1
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover
                r = rng.random(dim)
                jrand = rng.integers(0, dim)
                trial = np.where(r < CR, v, population[i])
                trial[jrand] = v[jrand]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                evals_used += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            generation += 1
            
            # Adaptive F and CR based on generation
            progress = min(generation / 50.0, 1.0)
            F = 0.8 - 0.3 * progress
            CR = 0.9 - 0.3 * progress
            
            # If F becomes too small, switch to local refinement
            if F < 0.15:
                break
        
        # Phase 2: Local refinement with Nelder-Mead style search
        remaining = budget - evals_used
        if remaining > 10 and time.time() < deadline and best_x is not None:
            x_local = best_x.copy()
            f_local = best_f
            
            # Adaptive step size starting from current scale
            step_size = 0.5
            min_step = 1e-6
            max_iter = min(remaining, 200)
            
            for _ in range(max_iter):
                if time.time() > deadline or evals_used >= budget:
                    break
                
                # Try moving in random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                x_new = x_local + step_size * direction
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    step_size = min(step_size * 1.3, 2.0)
                else:
                    step_size *= 0.7
                
                if step_size < min_step:
                    break
        
        return best_x, best_f
