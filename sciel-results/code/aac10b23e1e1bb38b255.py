"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES portfolio with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        self.dim = self.dim
        self.budget = self.budget
        calls_made = 0
        max_calls = self.budget

        # Try several diverse starting points with (1+1)-ES
        # Use a portfolio approach: allocate budget among multiple restarts
        n_restarts = 5
        calls_per_restart = max(1, (self.budget - 1) // n_restarts)
        
        best_x = None
        best_fx = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            if calls_made >= max_calls:
                break
                
            # Different initialization strategies
            if restart == 0:
                x = np.zeros(self.dim)
            elif restart == 1:
                x = rng.uniform(-5, 5, self.dim)
            elif restart == 2:
                x = rng.uniform(-3, 3, self.dim)
            elif restart == 3:
                x = rng.uniform(-2, 2, self.dim)
            else:
                x = np.zeros(self.dim) + rng.normal(0, 0.5, self.dim)
            
            x = np.clip(x, -5, 5)
            
            if calls_made >= max_calls:
                break
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            # (1+1)-ES for this restart
            sigma = 2.0
            for step in range(calls_per_restart):
                if time.time() > deadline - 10.0:
                    break
                if calls_made >= max_calls:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
            
            # Refine best point with smaller steps
            if best_x is not None and calls_made < max_calls and time.time() < deadline - 10.0:
                x = best_x.copy()
                fx = best_fx
                sigma = 0.5
                refine_steps = min(20, max_calls - calls_made)
                for _ in range(refine_steps):
                    if time.time() > deadline - 10.0:
                        break
                    if calls_made >= max_calls:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        sigma *= 0.8
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)
            calls_made += 1
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
