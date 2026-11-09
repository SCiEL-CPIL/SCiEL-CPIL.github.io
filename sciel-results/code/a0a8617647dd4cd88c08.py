"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        
        # Allocate budget among restarts
        # For f21 (Gallagher 101 peaks), multiple restarts help escape local optima
        n_restarts = 5
        calls_per_restart = self.budget // n_restarts
        total_calls = 0
        best_x = None
        best_fx = np.inf
        
        for restart in range(n_restarts):
            # Check time
            if time.time() > deadline - 10.0:
                break
            
            # Initialize at a different point
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            total_calls += 1
            
            if best_x is None or fx < best_fx:
                best_x, best_fx = x.copy(), fx
            
            sigma = 1.5
            # Run (1+1)-ES for this restart
            for _ in range(calls_per_restart - 1):
                if time.time() > deadline:
                    return best_x, best_fx
                if total_calls >= self.budget:
                    return best_x, best_fx
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.85
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
