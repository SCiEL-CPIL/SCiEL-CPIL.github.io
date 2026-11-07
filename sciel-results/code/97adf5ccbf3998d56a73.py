import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 295.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        evals = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Multi-start differential evolution
        # Population size for DE
        pop_size = 10
        # Number of restarts
        max_restarts = 5
        # DE parameters
        F = 0.5  # Mutation scale factor
        CR = 0.9  # Crossover rate
        
        def run_de(start_x, start_f, remaining_budget, rng_local):
            nonlocal evals, best_x, best_f
            
            # Initialize population around start point
            pop = np.zeros((pop_size, n))
            fit = np.zeros(pop_size)
            
            # First individual is the start point
            pop[0] = start_x
            fit[0] = start_f
            evals += 1
            
            # Generate rest of population
            for i in range(1, pop_size):
                if evals >= remaining_budget:
                    break
                x = start_x + rng_local.standard_normal(n) * 1.0
                x = np.clip(x, -5, 5)
                f = func(x)
                evals += 1
                pop[i] = x
                fit[i] = f
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Sort population
            idx = np.argsort(fit)
            pop = pop[idx]
            fit = fit[idx]
            
            # Update global best
            if fit[0] < best_f:
                best_f = fit[0]
                best_x = pop[0].copy()
            
            # DE loop
            stagnation_counter = 0
            last_best = best_f
            no_improve_generations = 0
            
            while evals + pop_size <= remaining_budget:
                if time.time() - t_start > max_time:
                    break
                
                # Check for convergence
                if no_improve_generations > 30:
                    break
                
                new_pop = np.zeros((pop_size, n))
                new_fit = np.zeros(pop_size)
                
                for i in range(pop_size):
                    if evals >= remaining_budget:
                        break
                    
                    # Differential mutation
                    if pop_size >= 3:
                        a, b, c = rng_local.choice(pop_size, size=3, replace=False)
                        mutant = pop[a] + F * (pop[b] - pop[c])
                    else:
                        mutant = pop[rng_local.choice(pop_size)] + F * rng_local.standard_normal(n)
                    
                    # Binomial crossover
                    trial = pop[i].copy()
                    j_rand = rng_local.integers(0, n)
                    for j in range(n):
                        if rng_local.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    
                    # Clip to bounds
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    evals += 1
                    
                    # Selection
                    if f_trial <= fit[i]:
                        new_pop[i] = trial
                        new_fit[i] = f_trial
                    else:
                        new_pop[i] = pop[i]
                        new_fit[i] = fit[i]
                
                # Sort new population
                idx = np.argsort(new_fit)
                new_pop = new_pop[idx]
                new_fit = new_fit[idx]
                
                # Update global best
                if new_fit[0] < best_f:
                    best_f = new_fit[0]
                    best_x = new_pop[0].copy()
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1
                
                pop = new_pop
                fit = new_fit
            
            return best_x, best_f
        
        # Multi-start strategy
        # Start 1: Center
        x_init = np.zeros(n)
        f_init = func(x_init)
        evals += 1
        best_x = x_init.copy()
        best_f = f_init
        
        # Run DE from center
        run_de(x_init, f_init, budget, rng)
        
        # Additional restarts from promising areas
        for restart in range(1, max_restarts):
            if evals >= budget:
                break
            if time.time() - t_start > max_time:
                break
            
            # Start near current best with some perturbation
            x_start = best_x + rng.standard_normal(n) * 0.5
            x_start = np.clip(x_start, -5, 5)
            f_start = func(x_start)
            evals += 1
            
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            
            run_de(x_start, f_start, budget, rng)
        
        return best_x, best_f
