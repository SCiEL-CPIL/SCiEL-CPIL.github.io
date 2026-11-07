"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule, diversified init, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # 10s safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Diversified Initialization
        # Evaluate a small set of diverse points to find a good starting location.
        # This helps avoid starting in a poor local minimum or a region with poor conditioning.
        n_init = 5
        candidates = []
        # Include the center of the box
        candidates.append(np.zeros(dim))
        # Add random points
        for _ in range(n_init - 1):
            candidates.append(rng.uniform(-5, 5, dim))
            
        for x0 in candidates:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        x = best_x.copy()
        fx = best_f

        # Phase 2: (1+1)-ES
        sigma = 2.0
        # We use a standard 1/5 rule approximation via multiplicative updates
        # Success: increase sigma, Failure: decrease sigma
        # Factors chosen to be robust
        inc_factor = 1.5
        dec_factor = 0.87

        while calls_made < budget:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= inc_factor
            else:
                sigma *= dec_factor
            
            # Clamp sigma to prevent numerical issues
            if sigma < 1e-4:
                sigma = 1e-4
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
# EVOLVE-BLOCK-END
