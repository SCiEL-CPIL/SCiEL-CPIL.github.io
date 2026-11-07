"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and time limit."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Explicit deadline guard: 300s limit, use 290s for safety
        start_time = time.time()
        deadline = start_time + 290
        
        best_x = None
        best_fx = float('inf')
        
        # Use multiple restarts to explore different regions
        n_restarts = 5
        calls_per_restart = max(1, (self.budget - 1) // n_restarts)
        
        for restart in range(n_restarts):
            if time.time() >= deadline:
                break
            
            # Initialize with a different strategy each restart
            if restart == 0:
                x = rng.uniform(-5, 5, self.dim)
            elif restart == 1:
                x = np.zeros(self.dim)
            elif restart == 2:
                x = rng.uniform(-2, 2, self.dim)
            elif restart == 3:
                x = rng.uniform(-3, 3, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            sigma = 2.0
            
            for _ in range(calls_per_restart):
                if time.time() >= deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                # If sigma becomes too small, restart within this segment
                if sigma < 0.01:
                    break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
