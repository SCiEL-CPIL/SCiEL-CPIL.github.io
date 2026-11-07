"""Multi-restart (1+1)-ES for BBOB f23 (Katsuura) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # Multi-restart strategy
        # Katsuura is multimodal. Single ES often gets stuck.
        # We divide the budget into chunks and restart the ES from random points.
        num_restarts = 5
        steps_per_restart = (self.budget - calls) // num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline or calls >= self.budget:
                break
            
            # Initialize new ES instance
            # Start from a random point to explore new basins
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            
            sigma = 1.0
            successes = 0
            window_size = 10
            steps_in_window = 0
            stall_count = 0
            stall_limit = 20
            
            # Run (1+1)-ES for this restart
            for step in range(steps_per_restart):
                if time.time() > deadline or calls >= self.budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                # Check for improvement
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    stall_count = 0
                    if fy < f_best:
                        f_best = fy
                        x_best = x.copy()
                else:
                    stall_count += 1
                
                # Adapt sigma (1/5 success rule)
                steps_in_window += 1
                if steps_in_window >= window_size:
                    if successes / window_size > 1/5:
                        sigma *= 1.1
                    else:
                        sigma /= 1.1
                    successes = 0
                    steps_in_window = 0
                    
                # Internal restart if stalled (optional, but helps within a phase)
                if stall_count >= stall_limit:
                    # Try a smaller step or just break out of this inner loop
                    # and let the outer loop handle the major restart
                    break

        # Final polishing step if budget remains
        if calls < self.budget and time.time() < deadline:
            sigma = 0.01
            x = x_best.copy()
            fx = f_best
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                calls += 1

        return x_best, f_best
# EVOLVE-BLOCK-END
