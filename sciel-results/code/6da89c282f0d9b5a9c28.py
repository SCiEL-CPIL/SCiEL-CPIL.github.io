"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: initial random sampling followed by (1+1)-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(0)
        
        # Phase 1: Initial random sampling to find a good starting point
        # Use 10% of budget for exploration, at least 5 points
        init_budget = max(5, int(budget * 0.1))
        
        best_x = None
        best_f = float('inf')
        
        # Evaluate origin as a strong prior for Schaffers F7
        x0 = np.zeros(n)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        for _ in range(init_budget - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES refinement
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Start with moderate step size
        
        remaining_budget = budget - init_budget
        
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                # Avoid sigma becoming too small
                if sigma < 1e-6:
                    sigma = 1e-6
        
        return best_x, best_f
# EVOLVE-BLOCK-END
