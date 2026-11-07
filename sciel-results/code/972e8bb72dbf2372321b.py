import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 250  # Leave 50s margin for 300s limit
        
        # Initialization
        # Start at 0, which is the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1
        
        # Initial step sizes
        # Start moderately large to cover the range [-5, 5] quickly
        steps = np.full(self.dim, 1.0)
        
        # Track best
        best_x = x.copy()
        best_fx = fx
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            x_old = x.copy()
            improved = False
            
            # Coordinate search
            for i in range(self.dim):
                # Try positive step
                x_trial = x.copy()
                x_trial[i] += steps[i]
                x_trial = np.clip(x_trial, -5, 5)
                
                # Only evaluate if it's a new point (avoid redundant calls if clipped to same)
                # But func clips anyway, so if x_trial == x, we can skip
                if not np.allclose(x_trial, x):
                    f_trial = func(x_trial)
                    calls += 1
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        steps[i] *= 2.0
                        improved = True
                        continue # Move to next dimension? Or stay? 
                        # Standard Hooke-Jeeves: if a step is accepted in direction i, 
                        # we typically do NOT try the negative direction in the same iteration 
                        # for that dimension, and we proceed to the next dimension.
                        # However, sometimes it's better to keep going in the same direction?
                        # No, standard is: accept, update step, next dim.
                    else:
                        # Try negative step
                        x_trial_neg = x.copy()
                        x_trial_neg[i] -= steps[i]
                        x_trial_neg = np.clip(x_trial_neg, -5, 5)
                        
                        if not np.allclose(x_trial_neg, x):
                            f_trial_neg = func(x_trial_neg)
                            calls += 1
                            if f_trial_neg < fx:
                                x = x_trial_neg
                                fx = f_trial_neg
                                steps[i] *= 2.0
                                improved = True
                            else:
                                steps[i] *= 0.5
                        else:
                            steps[i] *= 0.5
                else:
                    # Clipped to same point, reduce step
                    steps[i] *= 0.5
                    
            # Pattern Move
            if improved:
                # The vector of moves made in this cycle
                move_vec = x - x_old
                if np.any(move_vec != 0):
                    x_pattern = x + move_vec
                    x_pattern = np.clip(x_pattern, -5, 5)
                    
                    if not np.allclose(x_pattern, x):
                        f_pattern = func(x_pattern)
                        calls += 1
                        if f_pattern < fx:
                            x = x_pattern
                            fx = f_pattern
                        # If pattern move fails, we just stay at x (current best)
                        # We do NOT reset to x_old. We continue from x.
            
            # Update global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Stagnation check: if steps are too small, we might be stuck
            # But with budget 500, we just keep going until budget or time runs out.
            # If max step is < 1e-4, we can break early as we've converged.
            if np.max(steps) < 1e-4:
                break

        return best_x, best_fx
