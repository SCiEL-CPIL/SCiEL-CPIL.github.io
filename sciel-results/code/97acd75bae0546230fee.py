"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize
        x = np.random.uniform(-5, 5, n)
        fx = func(x)
        calls_made = 1
        
        sigma = 2.0
        successes = 0
        window = 15
        success_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * np.random.randn(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
            else:
                success_count = 0

            # Adapt sigma using 1/5 rule every 'window' steps
            # We check if we have a full window of data
            # Simple implementation: adjust every step based on recent history? 
            # Standard 1/5 rule: adjust every 15 steps.
            # Let's use a counter for the window.
            
            # To keep it simple and robust, we can adjust every step 
            # but that's not standard. Let's stick to the window approach.
            # We need a counter for the current window.
            
            # Actually, the simple version in the inspiration just used a loop.
            # Let's implement the standard 1/5 rule properly.
            
            # Since we don't have a window buffer in the simple loop above, 
            # let's restructure slightly to maintain a window or just use 
            # the simple heuristic from the inspiration which worked.
            
            # The inspiration code:
            # if fy < fx: sigma *= 1.5
            # else: sigma *= 0.87
            # This is a simplified 1/5 rule. Let's use that.
            
            if fy < fx:
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            sigma = np.clip(sigma, 1e-8, 1e8)

        return x, fx
# EVOLVE-BLOCK-END
