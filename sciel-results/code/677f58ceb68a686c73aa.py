"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        rng = np.random.default_rng(0)
        
        # Parameters
        sigma = 2.0
        stagnation_limit = 50  # Restart if no improvement for 50 steps
        
        while calls_made < total_budget and time.time() < deadline:
            # Initialize new search
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0
            no_improve_count = 0
            
            # Run (1+1)-ES until stagnation or budget/time limit
            while calls_made < total_budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve_count += 1
                
                if no_improve_count >= stagnation_limit:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
