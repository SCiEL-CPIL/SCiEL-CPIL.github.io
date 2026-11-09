"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive step size and restart mechanism."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Use a fixed seed for reproducibility within the instance, 
        # but allow different behavior if we restart? 
        # The task says "hidden seeded shift", implying the function is fixed.
        # Using a fixed RNG seed ensures we explore the same space structure.
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = evaluate(x)
        
        sigma = 2.0
        stagnation = 0
        max_stagnation = 30 # Restart if no improvement for 30 steps
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                # Improvement
                x, fx = y, fy
                sigma *= 1.5 # Increase step size on success
                stagnation = 0
            else:
                # No improvement
                x = np.clip(x + 0.5 * (y - x) * 0.1, -5, 5) # Small nudge? Or just keep x?
                # Standard (1+1) keeps x. Let's stick to keeping x to avoid wasting budget on bad moves.
                # Actually, standard is to just reject y.
                x = x # Keep parent
                sigma *= 0.87 # Decrease step size
                stagnation += 1
                
            # Restart Mechanism
            if stagnation > max_stagnation:
                # If stuck for too long, try to escape
                if sigma < 0.1:
                    # Step size too small, likely in a local min.
                    # Restart from best found with a larger step size
                    x = best_x.copy()
                    fx = best_f
                    sigma = 2.0
                    stagnation = 0
                else:
                    # Step size is still large but no progress. 
                    # Maybe just increase step size to jump over barriers?
                    sigma *= 1.2
                    stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
