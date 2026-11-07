import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        # CMA-ES implementation
        # Parameters
        dim = self.dim
        budget = self.budget
        
        # Initial mean
        x_mean = np.zeros(dim)
        # Initial sigma
        sigma = 2.0
        
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        if lambda_pop < 4:
            lambda_pop = 4
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        # We need to be careful with the budget.
        # Each generation uses lambda_pop evaluations.
        
        best_x = None
        best_f = np.inf
        
        # Helper to check time
        def check_time():
            return (time.time() - start_time) > (deadline - 5)
            
        # We will run generations until budget or time runs out
        # We need to handle the first evaluation of the initial mean?
        # Usually CMA-ES starts by evaluating the population.
        
        # Let's do a quick random search first to get a decent start?
        # Or just start CMA-ES from 0?
        # The problem has a shift. 0 might be far.
        # Let's do 5 random starts and pick the best one to initialize CMA-ES.
        
        # Phase 1: Random Search (10 evals)
        rng = np.random.default_rng(0)
        best_init_x = None
        best_init_f = np.inf
        for _ in range(10):
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            if f_try < best_init_f:
                best_init_f = f_try
                best_init_x = x_try
                
        x_mean = best_init_x
        sigma = 1.0 # Adjust initial sigma
        
        # Phase 2: CMA-ES
        # We have budget - 10 left.
        remaining_budget = budget - 10
        
        while remaining_budget >= lambda_pop:
            if check_time():
                break
                
            # Sample population
            pop_x = np.zeros((lambda_pop, dim))
            pop_f = np.zeros(lambda_pop)
            
            # Standard normal sampling
            # We need to handle the covariance matrix update properly
            # For simplicity and robustness, if full CMA-ES is too complex to code correctly in time,
            # we can use a simplified version or just use scipy's Nelder-Mead from the best random point.
            
            # Given the risk of bugs in a hand-coded CMA-ES, and the "Knowledge" hint about Nelder-Mead,
            # let's stick to a robust Multi-Start Nelder-Mead.
            
            # I will switch strategy to Multi-Start Nelder-Mead.
            # It is simpler and less prone to implementation errors.
            
            break # Exit CMA-ES loop, use Nelder-Mead strategy instead
            
        # Strategy: Multi-Start Nelder-Mead
        # We have ~490 budget left.
        # Let's do 10 Nelder-Mead runs, each with ~40 evals?
        # Or 5 runs with 80 evals?
        # Nelder-Mead options: maxiter, xatol, fatol.
        
        # We already have 10 random points.
        # Let's take the best 5 random points and run Nelder-Mead on them.
        
        # Collect random points
        random_points = []
        for _ in range(20): # 20 more random points
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            random_points.append((f_try, x_try))
            
        # Sort by fitness
        random_points.sort(key=lambda x: x[0])
        
        # Take top 5
        top_points = [x for f, x in random_points[:5]]
        
        # Run Nelder-Mead on each
        from scipy.optimize import minimize
        
        global_best_x = best_init_x
        global_best_f = best_init_f
        
        for x0 in top_points:
            if check_time():
                break
            try:
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
            except Exception:
                pass
        
        # Final Polish
        if not check_time():
            try:
                res = minimize(func, global_best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x
            except Exception:
                pass
                
        return global_best_x, global_best_f
