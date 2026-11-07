import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        budget = self.budget
        
        # State
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # 1. Cuckoo Search Phase
        # Population size
        pop_size = 20
        alpha = 0.01 # Step scaling
        
        # Initialize population
        rng = np.random.default_rng(42)
        # Start with a mix of random and zero-centered
        pop = np.random.uniform(-5, 5, (pop_size, n))
        # Ensure one is at 0
        pop[0] = 0.0
        
        f_pop = np.array([safe_call(p) for p in pop])
        # Handle None if budget/time ran out
        if f_pop[0] is None:
            return best_x, best_fx
            
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_fx = f_pop[best_idx]
        
        # Lévy flight step generation
        # Using a simplified Lévy flight: Normal distribution with occasional large steps
        # Or using scipy.stats.levy_stable if available, but it's slow/unstable.
        # Common approximation: 
        # step = rng.normal(0, 1, n) * rng.pareto(1.5) 
        
        max_cs_calls = int(budget * 0.7) # 70% budget for CS
        
        while calls_made < max_cs_calls and time.time() < deadline:
            for i in range(pop_size):
                if calls_made >= max_cs_calls or time.time() > deadline:
                    break
                
                # Lévy flight step
                # Step size distribution
                # u ~ N(0, sigma_u^2), v ~ N(0, sigma_v^2)
                # L = u / |v|^(1/beta)
                # Simplified: use rng.normal * rng.exponential or similar
                # Let's use a robust heavy-tailed distribution
                step = rng.normal(0, 1, n)
                # Multiply by a heavy-tailed scalar
                # Laplace is a good approximation for Lévy in some contexts
                scale = np.abs(rng.laplace(0, 1))
                step = step * scale * alpha
                
                new_x = pop[i] + step
                new_fx = safe_call(new_x)
                
                if new_fx is None:
                    break
                    
                if new_fx < f_pop[i]:
                    pop[i] = new_x
                    f_pop[i] = new_fx
                    if new_fx < best_fx:
                        best_fx = new_fx
                        best_x = new_x.copy()
                else:
                    # Cuckoo step: replace with a better solution from population
                    # Select a random better solution
                    better_indices = np.where(f_pop < f_pop[i])[0]
                    if len(better_indices) > 0:
                        # Select the best or a random better one
                        # Let's pick the global best of the current pop to accelerate convergence
                        # Or random better to maintain diversity. 
                        # For Discus, accelerating towards the best might be good.
                        # Let's use the current global best in the population.
                        global_best_idx = np.argmin(f_pop)
                        # With some probability, do this
                        if rng.random() < 0.7:
                            pop[i] = pop[global_best_idx].copy() + rng.normal(0, 0.1, n) # Small noise
                            f_pop[i] = safe_call(pop[i])
                            if f_pop[i] is None:
                                break
                            if f_pop[i] < best_fx:
                                best_fx = f_pop[i]
                                best_x = pop[i].copy()
        
        # 2. Nelder-Mead Phase
        # Use remaining budget for local refinement
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            # Wrap func to respect budget and time
            nm_calls = [0]
            def nm_func(x):
                if nm_calls[0] >= remaining_budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                nm_calls[0] += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_clipped.copy()
                return fx
            
            # Initial simplex around best_x
            # Default NM simplex is fine, but we can tune it
            # x0 = best_x
            # options: maxiter, ftol, xtol
            # maxiter should be limited to ensure we don't exceed budget
            # Each NM iteration uses ~n+1 calls.
            # max_iter = remaining_budget // (n + 1)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': int(remaining_budget // (n + 1)), 
                                        'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                pass
        
        return best_x, best_fx
