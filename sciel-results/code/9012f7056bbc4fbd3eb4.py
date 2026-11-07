import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Dense structured + random initialization
        # Katsuura is extremely multimodal; we need many diverse starts
        initial_points = []
        initial_points.append(np.zeros(dim))
        
        # Lattice-like structured points
        for i in range(3):
            for j in range(3):
                pt = np.zeros(dim)
                pt[0] = -4 + 4*i/2
                pt[1] = -4 + 4*j/2
                for k in range(2, dim):
                    pt[k] = rng.uniform(-4, 4)
                initial_points.append(pt)
        
        # Random in full domain
        for _ in range(10):
            initial_points.append(rng.uniform(-5, 5, dim))
        
        # Random in tighter domain
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 15.0:
                break
            x0 = clip(x0)
            evaluate(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: Differential Evolution (DE/rand/1) with population
        # This provides global exploration with population-level recombination
        pop_size = 20
        remaining = budget - calls_made
        if remaining > 100 and remaining_time() > 20.0:
            # Initialize population around best and randomly
            population = []
            fvals = []
            
            # Include current best
            population.append(best_x.copy())
            fvals.append(best_f)
            
            # Add diverse points
            for _ in range(pop_size - 1):
                if rng.random() < 0.4:
                    x = clip(best_x + 1.5 * rng.standard_normal(dim))
                elif rng.random() < 0.7:
                    x = clip(rng.uniform(-4, 4, dim))
                else:
                    x = clip(rng.uniform(-5, 5, dim))
                population.append(x)
                fvals.append(evaluate(x))
            
            # DE loop
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            max_generations = min(remaining // (pop_size + 1), 40)
            
            for gen in range(max_generations):
                if calls_made >= budget or remaining_time() < 10.0:
                    break
                
                for i in range(pop_size):
                    if calls_made >= budget or remaining_time() < 5.0:
                        break
                    
                    # Select three distinct individuals
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    
                    # Mutation: DE/rand/1
                    mutant = population[a] + F * (population[b] - population[c])
                    mutant = clip(mutant)
                    
                    # Crossover
                    trial = np.copy(population[i])
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    
                    trial = clip(trial)
                    f_trial = evaluate(trial)
                    
                    # Selection
                    if f_trial <= fvals[i]:
                        population[i] = trial
                        fvals[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                # Update best from population
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = population[idx_min].copy()
                
                # Occasionally inject diversity
                if gen % 5 == 4 and calls_made < budget - 10:
                    idx_worst = np.argmax(fvals)
                    if rng.random() < 0.3:
                        new_x = clip(rng.uniform(-5, 5, dim))
                    else:
                        new_x = clip(best_x + 2.0 * rng.standard_normal(dim))
                    population[idx_worst] = new_x
                    fvals[idx_worst] = evaluate(new_x)
                    if fvals[idx_worst] < best_f:
                        best_f = fvals[idx_worst]
                        best_x = new_x.copy()

        # Phase 3: Local refinement with Nelder-Mead from best and top candidates
        remaining = budget - calls_made
        if remaining > 20 and remaining_time() > 10.0:
            # Refine best point
            local_calls = [0]
            local_budget = min(remaining, 80)
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 1.0
                return evaluate(x)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': local_budget, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = clip(res.x)
            except Exception:
                pass

        # Phase 4: CMA-ES-like final polish with very small steps
        remaining = budget - calls_made
        if remaining > 5 and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            max_steps = min(remaining, 100)
            
            for _ in range(max_steps):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                
                if no_improve >= 20:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
