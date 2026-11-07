"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: global random restarts + (1+1)-ES for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        # Phase 1: Global exploration with random restarts
        # Büche-Rastrigin is highly multimodal; random restarts help escape local minima
        best_x = rng.uniform(-5, 5, self.dim)
        best_fx = func(best_x)
        
        # Use ~60% of budget for global search with restarts
        global_budget = int(self.budget * 0.6)
        remaining_budget = self.budget - 1  # already used 1 call
        
        # Each restart gets a short (1+1)-ES run
        restarts_per_phase = 15
        calls_per_restart = max(1, (global_budget) // restarts_per_phase)
        
        for _ in range(restarts_per_phase):
            if remaining_budget <= 0:
                break
            if time.time() > deadline:
                break
            
            # New random start
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            remaining_budget -= 1
            
            # Short (1+1)-ES from this point
            sigma = 1.0
            for _ in range(calls_per_restart):
                if remaining_budget <= 0:
                    break
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = max(0.05, min(sigma, 5.0))
            
            # Update global best
            if fx < best_fx:
                best_x, best_fx = x, fx
        
        # Phase 2: Refine the best point with a longer (1+1)-ES
        x, fx = best_x.copy(), best_fx
        sigma = 0.5
        
        while remaining_budget > 0:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            remaining_budget -= 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = max(0.001, min(sigma, 3.0))
            
            if fx < best_fx:
                best_x, best_fx = x, fx
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
