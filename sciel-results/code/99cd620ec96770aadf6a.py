"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule approximation and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        # Initial State
        # Start at the center of the box, which is a common prior for BBOB instances
        x = np.zeros(n)
        x = np.clip(x, -5, 5)
        fx = func(x)
        x_best = x.copy()
        f_best = fx
        
        # Step size
        sigma = 2.0
        
        # Simple success rate tracking for 1/5 rule
        # We use a window of the last 15 steps to adapt sigma
        # If success rate > 0.2, increase sigma. If < 0.2, decrease.
        # Inspiration 1 used a simpler multiplicative update which is faster.
        # Let's use the multiplicative update from Inspiration 1 as it scored 1.0
        # but ensure we handle bounds and time strictly.
        
        eval_count = 1
        
        while eval_count < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            try:
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
            except Exception:
                # If an exception occurs, we should stop to avoid invalid state
                # or return best found
                break
            
            eval_count += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                x_best = y.copy()
                f_best = fy
                # Increase step size
                # 1.5 is a reasonable multiplier for success in (1+1)-ES
                sigma *= 1.5
            else:
                # Failure
                # 0.87 is a reasonable multiplier for failure
                sigma *= 0.87
                
            # Prevent sigma from exploding or becoming too small
            if sigma > 100:
                sigma = 100
            if sigma < 1e-12:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
