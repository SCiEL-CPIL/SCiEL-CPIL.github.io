"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution with Nelder-Mead polish for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, total_calls
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Differential Evolution (DE/rand/1/bin)
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]^n
        population = np.random.uniform(-5, 5, (pop_size, n))
        fitness = np.array([evaluate(x) for x in population])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        remaining_calls = budget - total_calls
        stagnation_count = 0
        max_stagnation = 30  # Generations without improvement before restart

        while remaining_calls > 0:
            if time.time() > deadline:
                break

            # Check if we can do a full generation
            if remaining_calls < pop_size:
                break

            gen_best_before = best_f
            gen_calls = 0

            for i in range(pop_size):
                if remaining_calls - gen_calls < 1:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                idxs = np.random.choice(pop_size, 3, replace=False)
                a, b, c = idxs

                # Mutation: v = population[a] + F * (population[b] - population[c])
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5, 5)

                # Crossover: binomial crossover with rate CR
                r = np.random.rand()
                j_rand = np.random.randint(n)
                trial = population[i].copy()
                for j in range(n):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = evaluate(trial)
                gen_calls += 1

                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            remaining_calls -= gen_calls

            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_f:
                best_f = fitness[gen_best_idx]
                best_x = population[gen_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stagnating, inject diversity by replacing worst members with random points
            if stagnation_count >= max_stagnation:
                # Replace worst 5 members with random points
                worst_idx = np.argsort(fitness)[-5:]
                for idx in worst_idx:
                    population[idx] = np.random.uniform(-5, 5, n)
                    fitness[idx] = evaluate(population[idx])
                    remaining_calls -= 1
                stagnation_count = 0

            # Adapt F and CR slightly based on progress
            if stagnation_count == 0:
                F = max(0.5, F * 0.98)  # Reduce F on success
                CR = min(0.95, CR + 0.01)  # Increase CR on success
            else:
                F = min(1.0, F + 0.02)  # Increase F on stagnation
                CR = max(0.6, CR - 0.01)  # Decrease CR on stagnation

        # Phase 2: Nelder-Mead polish from the best point found
        # Use the remaining budget for local refinement
        if best_x is not None and remaining_calls > 10:
            # Allocate up to 50% of remaining budget for polish, but at least 10 calls
            polish_budget = min(remaining_calls, max(10, remaining_calls // 2))
            
            # Use Nelder-Mead with a tight tolerance
            # We need to track calls carefully
            polish_calls = [0]
            
            def polish_func(x):
                polish_calls[0] += 1
                return func(x)
            
            try:
                # Start from the best point found by DE
                x0 = best_x.copy()
                
                # Run Nelder-Mead with limited iterations
                # maxiter is not a direct call count, so we use a callback to stop
                result = minimize(
                    polish_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    },
                    callback=lambda x: None if polish_calls[0] < polish_budget else (_ for _ in ()).throw(Exception("stop"))
                )
                
                # Update best if Nelder-Mead found something better
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                # If Nelder-Mead raises an exception (e.g., from our stop mechanism),
                # we just continue with what we have
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
