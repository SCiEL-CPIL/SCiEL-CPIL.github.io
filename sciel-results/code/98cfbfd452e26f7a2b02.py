"""Rewritten program using coordinate-descent hybrid with adaptive steps."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Coordinate-descent hybrid with adaptive per-dimension step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at box center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension, initialized moderately
        steps = np.full(dim, 1.0)

        # Count function calls made so far
        calls = 1

        def try_point(x_trial):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return
            x_trial = np.clip(x_trial, -5.0, 5.0)
            f_trial = func(x_trial)
            calls += 1
            if f_trial < best_fx:
                best_fx = f_trial
                best_x = x_trial.copy()

        # Phase 1: Coordinate descent with adaptive step sizes
        # Alternate between line search along each axis and random perturbations
        axis_order = list(range(dim))
        
        for it in range(budget - 1):
            if calls >= budget:
                break
            
            # Choose an axis to explore (cyclic with some randomness)
            if it % dim == 0:
                axis = it % dim
            else:
                axis = rng.integers(0, dim)
            
            # Try positive and negative directions along this axis
            step = steps[axis]
            
            # Line search: try a few step sizes
            for factor in [1.0, 0.5, 0.25]:
                if calls >= budget:
                    break
                x_trial = x.copy()
                x_trial[axis] += step * factor
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    steps[axis] *= 1.3  # Increase step if improvement
                    if f_trial < best_fx:
                        best_fx = f_trial
                        best_x = x.copy()
                    break
                else:
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[axis] -= step * factor
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        steps[axis] *= 1.3
                        if f_trial < best_fx:
                            best_fx = f_trial
                            best_x = x.copy()
                        break
                    else:
                        steps[axis] *= 0.8  # Decrease step if no improvement
            
            # Periodic random perturbation to escape local minima
            if it % 10 == 5 and calls < budget:
                perturb = rng.standard_normal(dim) * 0.5
                x_trial = np.clip(x + perturb, -5.0, 5.0)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    if f_trial < best_fx:
                        best_fx = f_trial
                        best_x = x.copy()
            
            # Ensure steps don't get too small or too large
            steps = np.clip(steps, 0.01, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
