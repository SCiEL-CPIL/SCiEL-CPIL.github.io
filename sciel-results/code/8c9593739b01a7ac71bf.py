import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Initial sampling: Try 3 random points to find a good starting basin
        # This is cheap relative to the 500 budget
        best_x = None
        best_f = np.inf
        for _ in range(3):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # If we ran out of budget during initialization (unlikely), return
        if best_x is None:
             # Should not happen with budget 500
             return np.zeros(self.dim), np.inf

        x = best_x
        fx = best_f
        sigma = 2.0
        
        calls_made = 3
        stall_counter = 0
        min_sigma = 1e-5
        max_sigma = 10.0
        
        while calls_made < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate mutant
            step = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * step, -5, 5)
            
            # Evaluate
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                # Increase step size, but cap it
                sigma *= 1.5
                if sigma > max_sigma:
                    sigma = max_sigma
                stall_counter = 0
            else:
                # Failure
                sigma *= 0.8
                stall_counter += 1
                
                # Stall-escape logic:
                # If we have been failing for a while and the step size is very small,
                # it's likely we are stuck in a local precision limit or a flat region.
                # Reset step size to a moderate value to explore again.
                if stall_counter > 20 and sigma < 1e-3:
                    sigma = 1.0
                    stall_counter = 0
                elif stall_counter > 50:
                    # Even if sigma isn't tiny, if we are stuck for a very long time,
                    # try a larger kick
                    sigma = max(sigma * 2.0, 0.5)
                    stall_counter = 0

        return x, fx
