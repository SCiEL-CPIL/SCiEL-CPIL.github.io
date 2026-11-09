"""Multi-Start Local Search optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Local Search optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Partition budget into 30 restarts, 10 evaluations each
        n_restarts = 30
        evals_per_restart = 10
        
        # Initial step size for local search
        # Box is [-5, 5], width 10. A step of 1.0 is 10% of the box.
        initial_step = 1.0
        
        for r in range(n_restarts):
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # 1. Random Start
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # 2. Local Hill-Climb (Steepest Descent with fixed step)
            # We have 9 evaluations left in this restart budget (1 used for init)
            step_size = initial_step
            
            for i in range(9):
                if time.time() - start_time > time_limit:
                    return best_x, best_f
                
                # Generate a random direction
                direction = rng.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                
                # Try moving in the direction
                x_new = x + step_size * direction
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                
                if fx_new < fx:
                    # Move to new point
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    # Keep step size same or slightly increase if we are far from boundary?
                    # For simplicity in this "pure" strategy, keep fixed step per restart.
                    # However, if we hit a boundary, we might want to shrink.
                    # But the directive says "fixed-step-size ... without any global step-size adaptation".
                    # So we keep it fixed for the duration of the restart.
                else:
                    # No improvement. Try opposite direction?
                    # Or just stop this restart?
                    # With only 10 evals, trying opposite direction is expensive.
                    # A simple random direction hill climb often gets stuck.
                    # Let's try a slightly smaller step if we fail? 
                    # No, "fixed-step-size". 
                    # If we fail, we just stop the local search for this restart 
                    # to save budget for the next restart? 
                    # The directive says "dedicating exactly 10 evaluations to each".
                    # So we should use all 10.
                    # If we don't improve, we can try another random direction.
                    pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
