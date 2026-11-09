"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Adaptive Coordinate Descent (Powell-like) for BBOB f14.
    Combines the robustness of the seed's ES with the directionality of coordinate search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s/330s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        # Wrapper to count calls and clip to box
        def _eval(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x_clipped)

        # Initial point: Origin is often a good heuristic for shifted/rotated BBOB
        # as the shift is bounded [-4,4] and rotation preserves distance from 0.
        x = np.zeros(dim)
        fx = _eval(x)
        best_x = x.copy()
        best_fx = fx
        if not np.isfinite(fx):
            # Fallback if origin is invalid (shouldn't happen with clipping)
            x = np.random.uniform(-5, 5, dim)
            fx = _eval(x)
            best_x = x.copy()
            best_fx = fx

        # Adaptive Coordinate Descent
        # Initialize step sizes
        step_sizes = np.ones(dim) * 1.0
        
        no_improve_count = 0
        max_no_improve = 100 # Restart or shrink if stuck
        
        while calls < budget and time.time() < deadline:
            improved_in_pass = False
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_trial = x.copy()
                x_trial[d] += step_sizes[d]
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = _eval(x_trial)
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    improved_in_pass = True
                    step_sizes[d] *= 1.5 # Increase step if moving
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                    no_improve_count = 0
                else:
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] -= step_sizes[d]
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = _eval(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved_in_pass = True
                        step_sizes[d] *= 1.5 # Increase step if moving
                        if fx < best_fx:
                            best_x = x.copy()
                            best_fx = fx
                        no_improve_count = 0
                    else:
                        # No improvement in this coordinate
                        step_sizes[d] *= 0.8 # Shrink step
                        if step_sizes[d] < 1e-6:
                            step_sizes[d] = 1e-6

            if not improved_in_pass:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # If stuck, try a random perturbation or restart step sizes
                    # A small random kick can escape local minima
                    if calls + dim < budget:
                        x_kick = x + np.random.normal(0, 0.1, dim)
                        x_kick = np.clip(x_kick, -5.0, 5.0)
                        f_kick = _eval(x_kick)
                        if f_kick < fx:
                            x = x_kick
                            fx = f_kick
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx
                        # Reset step sizes to a moderate value to explore
                        step_sizes = np.ones(dim) * 0.5
                        no_improve_count = 0

        # Final Polish: Use remaining budget for very small steps around best_x
        # This helps with the "sin" terms which have fine structure
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            # Perform a few passes of very fine coordinate search
            fine_steps = [0.01, 0.001, 0.0001]
            for step in fine_steps:
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = _eval(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
