"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill
        rng = np.random.default_rng(42)
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0
        calls_used = 1
        stall_count = 0
        stall_threshold = 30
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                stall_count = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.85
                stall_count += 1
            
            # If stuck for a while, try a restart with smaller sigma
            if stall_count >= stall_threshold:
                # Try a few random restarts around current best
                for _ in range(3):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    y2 = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    fy2 = func(y2)
                    calls_used += 1
                    if fy2 < best_fx:
                        x, fx = y2, fy2
                        best_x, best_fx = x.copy(), fx
                        sigma = 1.0
                        stall_count = 0
                        break
                else:
                    # No improvement from restarts, shrink sigma more aggressively
                    sigma *= 0.5
                    stall_count = 0
            
            # Final polish: if sigma is very small, do fine-grained search
            if sigma < 0.01:
                for _ in range(5):
                    if calls_used >= self.budget or time.time() > deadline:
                        break
                    y3 = np.clip(best_x + 0.01 * rng.standard_normal(self.dim), -5, 5)
                    fy3 = func(y3)
                    calls_used += 1
                    if fy3 < best_fx:
                        best_x, best_fx = y3.copy(), fy3
                        x, fx = y3, fy3
                if sigma < 0.001:
                    break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
