import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # Safety margin against 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def time_check():
            return (time.time() - start_time) > timeout

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return None
            if time_check():
                return None
            
            # Clip to box to ensure valid input, though func also clips
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            eval_count += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- Initialization ---
        # Start with a random point
        x0 = rng.uniform(-5.0, 5.0, dim)
        f0 = evaluate(x0)
        if f0 is None:
            return np.zeros(dim), 0.0
            
        current_x = x0.copy()
        current_f = f0
        
        # Strategy Parameters
        # Initial step size: large enough to explore the box [-5, 5]^5
        # For dim=5, a step of 2.0 allows covering most of the space in ~5-10 steps
        sigma = 2.0 
        min_sigma = 0.01
        max_sigma = 5.0
        
        # Stagnation handling
        no_improve_count = 0
        stagnation_threshold = 20 # If no improvement in 20 steps, reduce sigma or restart
        
        # (1+1)-ES Main Loop
        while eval_count < budget and not time_check():
            # Generate candidate
            # Use standard normal perturbation
            step = rng.normal(0, 1, dim)
            x_new = current_x + sigma * step
            
            f_new = evaluate(x_new)
            if f_new is None:
                break
            
            if f_new < current_f - 1e-12:
                # Improvement
                current_x = x_new.copy()
                current_f = f_new
                no_improve_count = 0
                
                # Optional: Slightly increase sigma if we just found a big improvement?
                # Usually, in (1+1) ES, we keep sigma constant or adapt it slowly.
                # Here, we keep it constant unless stagnating.
                
            else:
                # No improvement
                no_improve_count += 1
                
                if no_improve_count >= stagnation_threshold:
                    # Strategy: Reduce step size to fine-tune, or if sigma is already small, try a random jump
                    if sigma > min_sigma * 2:
                        # Shrink sigma by a factor
                        sigma *= 0.7
                        no_improve_count = 0
                    else:
                        # If sigma is very small, we might be stuck in a local minimum or flat region.
                        # Perform a random restart from a different location?
                        # Or just a large random jump from current best?
                        # Let's try a large jump:
                        jump_size = 3.0
                        x_jump = current_x + rng.normal(0, 1, dim) * jump_size
                        x_jump = np.clip(x_jump, -5.0, 5.0)
                        f_jump = evaluate(x_jump)
                        if f_jump is None:
                            break
                        
                        if f_jump < current_f:
                            current_x = x_jump.copy()
                            current_f = f_jump
                            sigma = 2.0 # Reset sigma for new region
                        else:
                            # If jump didn't help, try shrinking further or just keep searching locally
                            # To avoid infinite loop of useless jumps, let's reduce threshold slightly
                            # or just continue with small sigma.
                            # Let's reset the counter to prevent immediate re-jump
                            no_improve_count = 0
                            # Try a moderate jump if we are really stuck
                            if no_improve_count == 0 and eval_count < budget - 10:
                                 # Actually, let's just let it run with small sigma for a bit longer
                                 pass

        return best_x, best_f
