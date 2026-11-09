"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Global Differential Evolution + CMA-ES Refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # DE is robust for multimodal functions. 
        # We allocate ~60% of the budget to DE to find the correct basin.
        # Budget: 500. DE budget: 300.
        de_budget = int(self.budget * 0.6)
        
        # DE parameters
        # popsize: 15 (multiplier of dim=5 -> 75 individuals? No, popsize in scipy DE is multiplier of N. 
        # Actually scipy DE popsize is a float, population size = int(popsize * N). 
        # Let's use popsize=10 -> 50 individuals.
        # maxiter: We need to limit function evaluations.
        # Each iteration evaluates popsize*N points. 
        # 50 evals per iter. 300 budget / 50 = 6 iterations. This is very low.
        # Let's use a smaller population to allow more iterations.
        # popsize=5 -> 25 individuals. 300/25 = 12 iterations. Still low.
        # popsize=3 -> 15 individuals. 300/15 = 20 iterations. Better.
        # However, DE with small popsize might converge prematurely or miss basins.
        # Alternative: Use a custom DE loop? No, stick to scipy for robustness unless necessary.
        # Let's try popsize=5 and see. If it's too slow, we might hit time limit.
        # Actually, 500 calls is small. 
        # Let's use a simpler global search: Random Sampling + Local Refinement?
        # The current program does Random + NM. 
        # The insight says DE + CMA-ES is better.
        # Let's try to make DE work with the budget.
        
        # We can also just do a large number of random samples and then CMA-ES on the best few?
        # But DE is guided.
        
        # Let's configure DE to use exactly the budget we give it.
        # scipy DE does not have a strict 'maxfev' option in all versions, but it respects maxiter.
        # We will wrap it to stop if budget is exceeded, but DE might overshoot.
        # To be safe, let's implement a simple custom DE or use the wrapper to return inf.
        # If DE gets inf, it might fail.
        
        # Let's try a different split: 
        # 1. 100 Random Samples (100 calls)
        # 2. 5 CMA-ES restarts from the best 5 random points (400 calls)
        # This is similar to the current one but using CMA-ES instead of NM.
        # CMA-ES is generally better than NM for 5D continuous optimization.
        
        # Let's stick to the DE insight but be careful with budget.
        # I will implement a simple manual DE loop to strictly control budget.
        
        # --- Phase 1: Differential Evolution (Manual Implementation for Budget Control) ---
        # Population size: 20
        # Iterations: ~10 (200 evals)
        # Total DE evals: 200
        
        pop_size = 20
        de_iterations = 10
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Use Halton or just random? Random is fine for 5D.
        pop = rng.uniform(-5, 5, (pop_size, self.dim))
        pop_f = np.array([wrapped_func(p) for p in pop])
        
        # Keep track of best
        best_idx = np.argmin(pop_f)
        current_best_x = pop[best_idx].copy()
        current_best_f = pop_f[best_idx]
        
        for it in range(de_iterations):
            if time.time() > deadline - 30:  # Leave 30s for CMA-ES
                break
            if call_count >= self.budget - 200: # Leave 200 for CMA-ES
                break
                
            for i in range(pop_size):
                if time.time() > deadline - 30:
                    break
                if call_count >= self.budget - 200:
                    break
                    
                # Select 3 distinct individuals
                idx = np.random.choice(pop_size, 3, replace=False)
                a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                
                # Mutation
                v = pop[i] + F * (a - b) + 0.1 * (c - pop[i]) # Slight self-attraction? No, standard is a-b.
                v = pop[i] + F * (a - b)
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                trial_f = wrapped_func(trial)
                
                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < current_best_f:
                        current_best_f = trial_f
                        current_best_x = trial.copy()
        
        best_x = current_best_x
        best_f = current_best_f
        
        # --- Phase 2: CMA-ES Refinement ---
        # Use remaining budget (~200-300 calls) to refine the best point found by DE.
        # CMA-ES is efficient in low dimensions.
        
        remaining_budget = self.budget - call_count
        if remaining_budget > 50:
            # CMA-ES population size for dim=5 is ~28.
            # 200 calls / 28 ~ 7 iterations.
            # Let's run CMA-ES with maxiter=10 to be safe, wrapper will stop it.
            
            # We need to pass the current best as x0.
            # CMA-ES in scipy uses 'maxiter'.
            # We also want to ensure it doesn't run out of time.
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='CMA', 
                    options={
                        'maxiter': 10, 
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'seed': 42
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Final check
        if time.time() > deadline:
            return best_x, best_f
            
        return best_x, best_f

# EVOLVE-BLOCK-END
