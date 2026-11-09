"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 320s limit
        
        # Use a fixed seed for reproducibility within a single run
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Adaptive step size
        sigma = 2.0
        success_window = 5
        successes = 0
        
        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                # Increase step size on success
                sigma *= 1.2
            else:
                sigma *= 0.9
                # If we fail, keep x but update best if needed
                pass
            
            # Update global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Adaptive sigma adjustment based on success rate (1/5 rule variant)
            # Simple version: if we've had too many successes in a row, reduce sigma slightly
            # to avoid overshooting, but the multiplicative update above handles this.
            
            # Additional safeguard: if sigma becomes too small, reset it
            if sigma < 1e-4:
                sigma = 0.1

        return best_x, best_fx
# EVOLVE-BLOCK-END
