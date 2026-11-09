"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: Global random search followed by local (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # Use 30% of budget for global search
        global_budget = int(self.budget * 0.3)
        if global_budget < 10:
            global_budget = 10
            
        x_best = None
        f_best = float('inf')
        
        for _ in range(global_budget):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x
        
        # Phase 2: Local Refinement
        # Use remaining budget for (1+1)-ES
        local_budget = self.budget - global_budget
        
        x = x_best.copy()
        fx = f_best
        sigma = 0.5  # Start with a smaller step size as we are in a basin
        
        for _ in range(local_budget):
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1  # Expand step size on success
            else:
                sigma *= 0.9  # Shrink step size on failure
                
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 5.0:
                sigma = 5.0

        return x, fx
# EVOLVE-BLOCK-END
