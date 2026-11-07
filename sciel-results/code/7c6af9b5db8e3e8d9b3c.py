"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-like adaptive step, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Initial evaluation
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            best_x = x.copy()
            best_f = fx
            
            sigma = 2.0
            no_improve = 0
            max_no_improve = 50
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.05, 10.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.95, 0.01)
                    no_improve += 1
                    
                if no_improve >= max_no_improve:
                    # Restart from best_x with larger sigma
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 2.0
                    no_improve = 0
                    
        except Exception:
            pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
