"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        import time
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Use system entropy for better diversity across instances/runs
        rng = np.random.default_rng()
        
        # Global best
        best_x = rng.uniform(-5, 5, self.dim)
        best_f = func(best_x)
        
        # Current search state
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        stall_count = 0
        max_stall = 50  # Restart if no improvement for 50 steps
        
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Improvement in local search
                x, fx = y, fy
                sigma *= 1.2 # More conservative increase
                stall_count = 0
                
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # No improvement
                sigma *= 0.9 # More conservative decrease
                stall_count += 1
                
                if stall_count > max_stall:
                    # Restart local search from a new random point
                    # But keep the global best
                    x = rng.uniform(-5, 5, self.dim)
                    fx = func(x)
                    calls_made += 1
                    sigma = 2.0
                    stall_count = 0
                    
                    # If the new point is better, update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
