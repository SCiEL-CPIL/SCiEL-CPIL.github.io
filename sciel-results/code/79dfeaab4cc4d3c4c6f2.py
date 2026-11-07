"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Strategy: multiple short restarts to handle hidden shifts
        # Each restart uses a portion of the budget
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        for restart in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize: for first restart, use center; for others, perturb around best
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                if best_x is not None:
                    x = best_x + rng.normal(0, 1.0, self.dim)
                else:
                    x = rng.uniform(-5, 5, self.dim)
            
            x = np.clip(x, -5, 5)
            
            if calls_made >= self.budget:
                break
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0
            # Adaptive parameters
            success_counter = 0
            total_steps = 0
            window = 10  # window for 1/5 rule
            
            for _ in range(budget_per_restart - 1):
                # Check time
                if time.time() > deadline:
                    break
                
                if calls_made >= self.budget:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_counter += 1
                    sigma *= 1.05  # small increase
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95  # small decrease
                
                total_steps += 1
                
                # Adaptive step size based on success rate
                if total_steps >= window:
                    success_rate = success_counter / total_steps
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    success_counter = 0
                    total_steps = 0
                
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
