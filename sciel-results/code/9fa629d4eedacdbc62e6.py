"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Randomized Coordinate Descent with Adaptive Steps and Restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42) # Keep seed consistent for reproducibility in testing, but logic is different

        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step sizes
        steps = np.full(dim, 2.0)
        
        calls_used = 1
        no_improve_global = 0
        min_step = 1e-3
        max_step = 5.0
        
        dim_order = list(range(dim))
        current_dim_idx = 0

        while calls_used < budget:
            if current_dim_idx >= dim:
                rng.shuffle(dim_order)
                current_dim_idx = 0
            d = dim_order[current_dim_idx]
            current_dim_idx += 1

            step = steps[d]
            
            # 1. Try positive direction
            if calls_used < budget:
                trial_pos = x.copy()
                trial_pos[d] += step
                trial_pos = np.clip(trial_pos, -5, 5)
                f_pos = func(trial_pos)
                calls_used += 1
                
                if f_pos < fx:
                    x = trial_pos
                    fx = f_pos
                    steps[d] = min(step * 1.2, max_step)
                    # Update best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve_global = 0
                    else:
                        no_improve_global += 1
                    continue # Next dimension

            # 2. Try negative direction
            if calls_used < budget:
                trial_neg = x.copy()
                trial_neg[d] -= step
                trial_neg = np.clip(trial_neg, -5, 5)
                f_neg = func(trial_neg)
                calls_used += 1
                
                if f_neg < fx:
                    x = trial_neg
                    fx = f_neg
                    steps[d] = min(step * 1.2, max_step)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve_global = 0
                    else:
                        no_improve_global += 1
                    continue # Next dimension
                else:
                    # No improvement, shrink step
                    steps[d] = max(step * 0.7, min_step)
                    
                    # 3. Escape mechanism if step is tiny
                    if steps[d] <= min_step * 1.1:
                        if calls_used < budget:
                            # Random jump
                            perturbation = rng.standard_normal(dim) * 1.0
                            trial_jump = x + perturbation
                            trial_jump = np.clip(trial_jump, -5, 5)
                            f_jump = func(trial_jump)
                            calls_used += 1
                            
                            if f_jump < fx:
                                x = trial_jump
                                fx = f_jump
                                steps = np.full(dim, rng.uniform(1.0, 2.0))
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                    no_improve_global = 0
                            else:
                                no_improve_global += 1
                        else:
                            no_improve_global += 1
                    else:
                        no_improve_global += 1

            # Restart mechanism
            if no_improve_global > 40:
                x = best_x.copy()
                fx = best_fx
                steps = np.full(dim, rng.uniform(1.0, 3.0))
                no_improve_global = 0
                rng.shuffle(dim_order)
                current_dim_idx = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
