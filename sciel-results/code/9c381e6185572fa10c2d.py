"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and a final local refinement phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Safety margin: 290s is well within the 330s kill limit.
        deadline = time.time() + 290.0
        rng = np.random.default_rng(0)
        
        # Initialize with a random point in the box
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Parameters from Inspiration 1 (Score 1.0)
        # sigma=2.0 is a good initial scale for [-5, 5]^5
        sigma = 2.0
        
        calls_made = 1
        # Reserve a small number of calls for final local refinement.
        # This refinement is deterministic and only runs if we have time and budget left.
        reserve_calls = 20
        max_main_calls = self.budget - reserve_calls
        
        for i in range(max_main_calls - 1):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Using standard normal noise scaled by sigma
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            # Update step size and position
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5  # Aggressive expansion on success
            else:
                sigma *= 0.87 # Moderate contraction on failure
            
            # Track best explicitly
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
                
        # Final local refinement phase:
        # If we have remaining budget and time, do a simple coordinate descent
        # to fine-tune the best solution found.
        if calls_made < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            # Start with a small step size for coordinate descent
            step = 0.1
            # Try to improve by moving along each coordinate
            for _ in range(5):  # 5 passes of coordinate descent
                if calls_made >= self.budget or time.time() > deadline:
                    break
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    # Try positive step
                    y = x.copy()
                    y[d] = np.clip(y[d] + step, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                    # Try negative step
                    if calls_made < self.budget and time.time() < deadline:
                        y = x.copy()
                        y[d] = np.clip(y[d] - step, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx
                # Reduce step size for next pass
                step *= 0.5
                if step < 1e-6:
                    break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
