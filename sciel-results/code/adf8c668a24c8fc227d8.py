"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        def time_ok():
            return time.time() < deadline
        
        # Phase 1: Main exploration with (1+1)-ES
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 3.0
        best_x, best_fx = x.copy(), fx
        no_improve_count = 0
        max_no_improve = 50
        
        calls_made = 1
        while calls_made < self.budget - 20:  # reserve 20 for polish
            if not time_ok():
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                no_improve_count = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.85
                no_improve_count += 1
                
                # Restart if stuck too long
                if no_improve_count >= max_no_improve:
                    # Perturb from best found so far
                    x = np.clip(best_x + 1.5 * rng.standard_normal(self.dim), -5, 5)
                    fx = func(x)
                    calls_made += 1
                    sigma = 1.0
                    no_improve_count = 0
        
        # Phase 2: High-precision polish pass
        if time_ok():
            x, fx = best_x.copy(), best_fx
            sigma = 0.05
            polish_budget = min(20, self.budget - calls_made)
            for _ in range(polish_budget):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                    sigma *= 0.9
                else:
                    sigma *= 0.7
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
