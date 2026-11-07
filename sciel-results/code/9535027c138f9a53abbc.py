import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        # Initialize with center point to ensure best_x is not None
        # This consumes 1 call
        x0 = np.zeros(self.dim)
        safe_func(x0)
        
        if self.calls >= self.budget or not self._check_time():
            return self.best_x, self.best_f

        # Strategy: Island-Model Evolution Strategy (via Differential Evolution)
        # The Lunacek function is bi-modal. DE is a population-based method that
        # maintains diversity and is less prone to getting stuck in a single local
        # minimum compared to single-point methods like Nelder-Mead or (1+1)-ES.
        
        # We will use scipy.optimize.differential_evolution.
        # To fit the budget of ~499 remaining calls:
        # DE cost per generation = population_size
        # We want to run for a few generations.
        # Let's try a small population to allow more generations.
        # Pop size 10 -> 50 generations.
        # Pop size 20 -> 25 generations.
        
        # DE parameters:
        # - bounds: [(-5, 5)] * dim
        # - init: 'latinhypercube' for good initial spread
        # - maxiter: set high, we will stop via callback or budget
        # - tol: not used as primary stop, we rely on budget
        # - polish: True (uses Nelder-Mead at the end, which is good for final precision)
        
        # However, scipy DE does not easily expose the current best in a callback
        # in a way that allows us to stop immediately if we hit the budget, 
        # because it manages its own loop. 
        # But we can pass a maxiter that is an upper bound, and rely on the 
        # fact that each function call increments self.calls.
        # If we exceed budget, safe_func returns best_f and doesn't increment.
        # But DE might keep running its internal logic.
        
        # A safer approach:
        # We can implement a custom stopping criterion by wrapping the function
        # such that it raises an exception or returns a specific value if budget is hit,
        # but DE might catch exceptions.
        
        # Better approach:
        # Estimate maxiter.
        # Pop size P.
        # Total calls = P * maxiter + P (initial) + polish cost.
        # Let's use P=10.
        # Budget remaining = 499.
        # 499 / 10 = ~49 iterations.
        # Let's set maxiter=40 to be safe, leaving room for polish.
        
        # To ensure we don't blow the budget, we can use a smaller population
        # or fewer iterations.
        
        # Let's try a hybrid: 
        # 1. Run DE with a small population (e.g., 15) and maxiter=20.
        # 2. Cost: 15 * 20 = 300 calls.
        # 3. Polish: ~15-30 calls.
        # Total ~345 calls. We have ~150 left.
        # We can run a second DE run or a local search.
        
        # Actually, let's just run DE with a dynamic maxiter based on remaining budget.
        # But DE initializes the population first.
        
        # Let's use a fixed, conservative setting that fits well within 500 calls.
        # Pop size = 20.
        # Maxiter = 15.
        # Init cost = 20.
        # Iter cost = 20 * 15 = 300.
        # Polish cost = ~20.
        # Total = 340.
        # Remaining = 160.
        
        # We can use the remaining budget for a second, more targeted DE run
        # or a fine-grained local search.
        
        # Let's define the bounds
        bounds = [(-5.0, 5.0)] * self.dim
        
        # Run DE
        # We will catch any exceptions from DE
        try:
            # maxiter=15, popsize=20
            # init='latinhypercube'
            # seed=42 for reproducibility in the instance context? 
            # The task says "hidden seeded shift", but our optimizer instance is fresh.
            # Using a fixed seed ensures deterministic behavior per instance.
            
            # We need to make sure DE stops if we run out of budget.
            # Since safe_func returns best_f when budget is exceeded, DE will see
            # constant values and might converge or stall.
            # It's safer to limit maxiter strictly.
            
            res = so.differential_evolution(
                safe_func, 
                bounds, 
                maxiter=15, 
                popsize=20, 
                init='latinhypercube', 
                seed=42, 
                polish=True, 
                tol=1e-6
            )
            
            if res.success or res.x is not None:
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
                    
        except Exception:
            # If DE fails, we fall back to whatever we have
            pass

        # If we still have budget, do a final local polish using Nelder-Mead 
        # starting from the best point found so far.
        if self.best_x is not None and self.calls < self.budget - 10 and self._check_time():
            # Simple local search with Nelder-Mead
            try:
                res_nm = so.minimize(
                    safe_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = res_nm.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f
