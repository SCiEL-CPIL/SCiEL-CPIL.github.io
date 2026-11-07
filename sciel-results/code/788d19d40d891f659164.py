"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0  # Start with a moderate step size
        
        # 1/5 Success Rule Parameters
        window_size = 10
        success_count = 0
        total_in_window = 0
        
        calls_made = 1
        
        for i in range(self.budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            # Update best
            if fy < fx:
                x, fx = y, fy
                success_count += 1
            
            total_in_window += 1
            
            # Adjust sigma based on 1/5 rule
            if total_in_window >= window_size:
                success_rate = success_count / total_in_window
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                
                # Reset window
                success_count = 0
                total_in_window = 0
                
        return x, fx
# EVOLVE-BLOCK-END
