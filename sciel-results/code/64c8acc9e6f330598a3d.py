"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule approximation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Initial random sampling to find a good starting point
        # Use a small fraction of the budget for this, e.g., 10 calls
        n_init = min(10, budget)
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        calls_made = n_init
        
        # Phase 2: (1+1)-ES optimization
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # Adaptation factors for 1/5 success rule heuristic
        # If success rate is ~1/5, we want to adjust sigma accordingly.
        # Common heuristic: multiply by 1.2 on success, 0.8 on failure.
        up_factor = 1.2
        down_factor = 0.8
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x + sigma * rng.standard_normal(dim)
            # Clip to domain
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            # Update step size and best solution
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= up_factor
            else:
                # Failure
                sigma *= down_factor
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
