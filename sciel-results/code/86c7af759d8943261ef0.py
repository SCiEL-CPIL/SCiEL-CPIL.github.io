"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Differential Evolution (DE) based optimizer for BBOB f20 (Schwefel).
    
    Strategy:
    1. Initialize a diverse population.
    2. Run DE with 'best1bin' mutation and binomial crossover.
    3. Use the best individual found to seed a Nelder-Mead local search
       for fine-tuning, utilizing any remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # 5s safety margin before 330s kill
        
        dim = self.dim
        lo, hi = -5.0, 5.0
        budget = self.budget
        
        # State tracking
        calls_made = 0
        best_x_global = None
        best_f_global = float('inf')

        def call_func(x):
            nonlocal calls_made, best_x_global, best_f_global
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            calls_made += 1
            fx = func(x)
            
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
                
            return fx

        # --- Phase 1: Differential Evolution ---
        
        # Population size heuristic: typically 5-15 * dim
        # For dim=5, N=25 is a good balance between diversity and cost.
        N = 25
        if budget < 200:
            N = max(5, budget // 20)
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(lo, hi, (N, dim))
        
        # Evaluate initial population
        pop_f = np.array([call_func(x) for x in pop])
        
        # Sort initial population
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Number of generations we can afford
        # Each generation costs N evaluations.
        # Reserve ~20% of budget for local polish.
        max_gens = (budget * 0.8) // N
        
        for gen in range(max_gens):
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Current best in population is pop[0]
            best_in_pop = pop[0]
            
            for i in range(N):
                if calls_made >= budget or time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals different from i
                while True:
                    r1, r2, r3 = rng.choice(N, 3, replace=False)
                    if len({i, r1, r2, r3}) == 4:
                        break
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                # Using 'best1bin' variant: v = x_best + F * (x_r1 - x_r2)
                # This is often faster for convergence.
                v = best_in_pop + F * (pop[r1] - pop[r2])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover: binomial
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                f_trial = call_func(trial)
                
                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    
            # Re-sort population to keep best at index 0
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]

        # --- Phase 2: Local Polish (Nelder-Mead) ---
        
        # If we have budget left, try to refine the best point found.
        # Nelder-Mead is derivative-free and effective for smooth functions.
        if calls_made < budget and best_x_global is not None:
            
            # Limit the number of function calls for local search
            # to ensure we don't overshoot the total budget.
            # We check budget inside the callback.
            
            # Define a wrapper that counts calls and respects deadline
            def wrapped_func(x):
                return call_func(x)
            
            try:
                # Nelder-Mead parameters
                # maxiter is not strictly a call count, but we rely on budget checks.
                # We'll let it run until budget is exhausted or it converges.
                res = minimize(
                    wrapped_func, 
                    best_x_global, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, # Safety cap on iterations
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # The best point is tracked in best_x_global
            except (ValueError, TimeoutError):
                # Budget or time exhausted
                pass
            except Exception:
                # Catch any other unexpected errors to ensure we return valid state
                pass

        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
