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
        deadline = start_time + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls_made = 1
        
        # Track progress for adaptive sigma
        success_count = 0
        window_size = 10
        window = []
        
        for _ in range(self.budget - 1):
            # Check deadline before each evaluation
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                window.append(1)
            else:
                window.append(0)
                
            # Adapt sigma based on success rate in recent window
            if len(window) >= window_size:
                success_rate = sum(window[-window_size:]) / window_size
                # 1/5 rule: aim for 20% success rate
                if success_rate > 0.25:
                    sigma *= 1.2
                elif success_rate < 0.15:
                    sigma *= 0.8
                window = window[-window_size:]
            else:
                # Early on, use simpler adaptation
                if success_count > 0 and calls_made > 5:
                    recent_success = sum(window) / len(window) if window else 0
                    if recent_success > 0.25:
                        sigma *= 1.1
                    elif recent_success < 0.15:
                        sigma *= 0.9
                
        return x, fx
# EVOLVE-BLOCK-END
