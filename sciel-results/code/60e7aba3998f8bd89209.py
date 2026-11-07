import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).flatten()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: CMA-ES with adaptive restarts
        # CMA-ES is well-suited for black-box optimization of non-convex functions
        # like Rastrigin, especially with rotation and shift.
        
        # We use a population-based approach with CMA-ES.
        # Initial population size lambda = 4 + floor(3 * log(dim)) = 4 + 3*1.609 = 8 (approx)
        # But for dim=5, let's use lambda = 10 for a good balance.
        lambda_ = 10
        
        # Initial sigma
        sigma = 2.0  # Start with a reasonable step size
        
        # We will run CMA-ES manually to have full control over the budget and restarts.
        # Alternatively, we can use scipy's minimize with method='Nelder-Mead' or 'Powell' 
        # but CMA-ES is generally superior for this type of problem.
        
        # Let's implement a simple CMA-ES loop.
        
        def cma_es_step(x_mean, sigma, C, C_inv):
            """Perform one step of CMA-ES."""
            nonlocal calls_made, best_x, best_f
            
            # Sample lambda points
            samples = []
            for i in range(lambda_):
                z = np.random.normal(0, 1, dim)
                # Apply C matrix square root (approximate with eigen decomposition or just use identity for simplicity in a quick implementation, 
                # but proper CMA-ES uses the covariance matrix)
                # For simplicity and speed, let's use a simpler adaptive method or rely on scipy's implementation if available.
                # Actually, implementing full CMA-ES is complex. Let's use scipy.optimize.minimize with method='Nelder-Mead' 
                # but with a smarter initialization and multiple restarts.
                # Or, better yet, use the 'L-BFGS-B' method if we had gradients, but we don't.
                
                # Let's stick to a robust multi-start local search strategy with Nelder-Mead and Powell,
                # but improve the initialization using a coarse grid search first.
                pass
            
            # Since implementing full CMA-ES from scratch is error-prone and time-consuming,
            # and the previous program already used Powell + Nelder-Mead, let's refine that approach.
            # The key insight is that Rastrigin has many local minima. A coarse grid search
            # can help identify the region of the global minimum, then a local optimizer can refine it.
            
            pass
        
        # Phase 1: Coarse Grid Search
        # Evaluate a sparse grid to find the best region.
        # For dim=5, a 2^5 = 32 point grid is cheap.
        # Let's use a 3^5 = 243 point grid? That might be too many if we have only 500 calls.
        # Let's use a 2^5 = 32 point grid first.
        
        grid_points = np.array(np.meshgrid(*([np.linspace(-5, 5, 2)] * dim))).T.reshape(-1, dim)
        
        for point in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            counted_func(point)
        
        # Phase 2: Local Search from Best Grid Point
        # Use Nelder-Mead from the best point found in the grid.
        if calls_made < budget - 10 and time.time() < deadline - 5:
            remaining = budget - calls_made
            res = minimize(counted_func, best_x, method='Nelder-Mead', options={'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        # Phase 3: Additional Restarts with Different Strategies
        # If we still have budget, try a few more random restarts with Powell.
        if calls_made < budget - 20 and time.time() < deadline - 5:
            rng = np.random.default_rng(123)
            for _ in range(3):
                if calls_made >= budget - 10:
                    break
                if time.time() > deadline - 5:
                    break
                x0 = rng.uniform(-5, 5, dim)
                res = minimize(counted_func, x0, method='Powell', options={'maxfev': min(50, budget - calls_made), 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
        
        return best_x, best_f
