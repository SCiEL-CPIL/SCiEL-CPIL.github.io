"""Hybrid multi-start optimizer: Nelder-Mead refinement + Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer combining Nelder-Mead local refinement with Differential Evolution."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Structured + random initialization
        # Sample points in the known region [-4, 4]^5 where minimum lies
        init_points = []
        
        # Center
        init_points.append(np.zeros(d))
        
        # Structured grid-like points (sparser)
        for _ in range(10):
            init_points.append(rng.uniform(-4, 4, d))
        
        # Near-center focused
        for _ in range(15):
            init_points.append(rng.uniform(-2, 2, d))
        
        # Full range
        for _ in range(15):
            init_points.append(rng.uniform(-5, 5, d))
        
        # Evaluate initial points
        for x0 in init_points:
            if calls_made >= B or remaining_time() < 20.0:
                break
            evaluate(x0)

        if best_x is None:
            best_x = np.zeros(d)
            evaluate(best_x)

        # Phase 2: Nelder-Mead local refinement from best point
        # This exploits the smooth structure of Katsuura for rapid local convergence
        if calls_made < B and remaining_time() > 20.0:
            local_budget = min(B - calls_made, 120)
            local_calls = [0]
            
            def local_func(x):
                local_calls[0] += 1
                if local_calls[0] > local_budget:
                    return best_f + 100.0
                return evaluate(x)
            
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead',
                             options={'maxiter': local_budget, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'maxfev': local_budget})
            except Exception:
                pass

        # Phase 3: Differential Evolution for global exploration
        remaining_calls = B - calls_made
        if remaining_calls > 80 and remaining_time() > 25.0:
            pop_size = 40
            # Initialize population: mix of near-best, mid-range, and full range
            population = []
            fitnesses = []
            
            # Seed with best found
            population.append(best_x.copy())
            fitnesses.append(best_f)
            
            # Add diverse points
            for i in range(pop_size - 1):
                if i < 8:
                    # Near best - for local exploration
                    x = clip(best_x + 0.3 * rng.standard_normal(d))
                elif i < 20:
                    # Mid-range
                    x = clip(rng.uniform(-4, 4, d))
                else:
                    # Full range
                    x = clip(rng.uniform(-5, 5, d))
                population.append(x)
                fitnesses.append(evaluate(x))
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]
            
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0].copy()
            
            # DE/rand/1 loop
            F = 0.7  # scaling factor
            CR = 0.85  # crossover rate
            
            gen = 0
            stagnation_counter = 0
            prev_best = best_f
            
            while calls_made < B and remaining_time() > 5.0:
                gen += 1
                
                # Check for stagnation
                if best_f < prev_best - 1e-10:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                prev_best = best_f
                
                # If stagnating for many generations, inject random diversity
                if stagnation_counter > 10:
                    worst_idx = np.argmax(fitnesses)
                    new_point = rng.uniform(-5, 5, d)
                    fnew = evaluate(new_point)
                    population[worst_idx] = new_point
                    fitnesses[worst_idx] = fnew
                    if fnew < best_f:
                        best_f = fnew
                        best_x = new_point.copy()
                    stagnation_counter = 0
                
                # Generate offspring for each member
                for i in range(pop_size):
                    if calls_made >= B or remaining_time() < 3.0:
                        break
                    
                    # Select three distinct random indices different from i
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    
                    # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                    v = population[r1] + F * (population[r2] - population[r3])
                    v = clip(v)
                    
                    # Crossover: binary crossover with CR
                    trial = np.empty(d)
                    j_rand = rng.integers(0, d)
                    for j in range(d):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]
                    
                    trial = clip(trial)
                    f_trial = evaluate(trial)
                    
                    # Selection
                    if f_trial <= fitnesses[i]:
                        population[i] = trial
                        fitnesses[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                
                # Periodically re-inject best into population to prevent loss
                if gen % 3 == 0 and fitnesses[0] > best_f:
                    population[0] = best_x.copy()
                    fitnesses[0] = best_f
                    order = np.argsort(fitnesses)
                    population = [population[i] for i in order]
                    fitnesses = [fitnesses[i] for i in order]

        return best_x, best_f
# EVOLVE-BLOCK-END
