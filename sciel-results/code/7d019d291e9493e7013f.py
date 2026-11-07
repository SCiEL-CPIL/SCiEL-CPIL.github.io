"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Time guard: return best if we exceed 320s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 320.0

        # (1+1)-ES Parameters
        # Initial step size
        sigma = 1.0
        
        # 1/5 success rule parameters
        window_size = 10
        successes = 0
        target_rate = 0.2
        
        # Best tracking
        x = rng.uniform(-5, 5, dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls_made = 1
        
        # Restart logic
        # If no improvement in a certain number of steps, restart globally
        # But we need to be careful with budget.
        # Let's use a "stagnation" counter.
        # If stagnation > 50, restart x to a random point in the box.
        # This helps escape local minima on multimodal functions like Rastrigin/Gallagher.
        
        stagnation_count = 0
        # To avoid infinite loops of restarting without improvement, 
        # we can limit restarts or just let it run.
        # With 300 budget, we can afford several restarts.
        
        # We will restart if stagnation_count exceeds a threshold, say 30.
        # This uses 30 calls to realize we are stuck, then 1 call to reset.
        # This is a simple heuristic.
        
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Standard normal mutation
            z = rng.normal(0, 1, dim)
            x_new = x + sigma * z
            
            # Clip to bounds [-5, 5]
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            # Update 1/5 success rule
            if f_new <= f_x:
                successes += 1
                x = x_new
                f_x = f_new
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x.copy()
                    stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Adapt sigma every window_size steps
            if calls_made % window_size == 0:
                rate = successes / window_size
                if rate > target_rate:
                    sigma *= 1.25
                elif rate < target_rate:
                    sigma /= 1.25
                successes = 0
                
                # Ensure sigma doesn't explode or vanish
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Check for stagnation and restart if needed
                if stagnation_count > 30:
                    # Restart from a random point
                    # This is a "global restart"
                    x = rng.uniform(-5, 5, dim)
                    f_x = func(x)
                    calls_made += 1
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                    stagnation_count = 0
                    # Reset sigma to a moderate value for exploration
                    sigma = 1.0
                    successes = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
