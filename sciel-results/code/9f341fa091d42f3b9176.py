"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Coordinate-wise line search (Cyclic Coordinate Descent) for BBOB f14.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        # Start at origin
        x = np.zeros(dim)
        fx = func(x)
        calls_used += 1

        # Adaptive step sizes for each dimension
        steps = np.ones(dim)

        def line_search_1d(x, i, step, fx):
            nonlocal calls_used
            # Try positive
            x_trial = x.copy()
            x_trial[i] += step
            x_trial[i] = np.clip(x_trial[i], -5, 5)
            f_trial = func(x_trial)
            calls_used += 1
            
            if f_trial < fx:
                x_new = x_trial
                f_new = f_trial
                # Expand
                while not check_time():
                    x_trial_next = x_new.copy()
                    x_trial_next[i] += step
                    x_trial_next[i] = np.clip(x_trial_next[i], -5, 5)
                    f_trial_next = func(x_trial_next)
                    calls_used += 1
                    if f_trial_next < f_new:
                        x_new = x_trial_next
                        f_new = f_trial_next
                    else:
                        break
                return x_new, f_new, step
            else:
                # Try negative
                x_trial = x.copy()
                x_trial[i] -= step
                x_trial[i] = np.clip(x_trial[i], -5, 5)
                f_trial = func(x_trial)
                calls_used += 1
                if f_trial < fx:
                    x_new = x_trial
                    f_new = f_trial
                    # Expand
                    while not check_time():
                        x_trial_next = x_new.copy()
                        x_trial_next[i] -= step
                        x_trial_next[i] = np.clip(x_trial_next[i], -5, 5)
                        f_trial_next = func(x_trial_next)
                        calls_used += 1
                        if f_trial_next < f_new:
                            x_new = x_trial_next
                            f_new = f_trial_next
                        else:
                            break
                    return x_new, f_new, step
                else:
                    # Shrink step if no improvement
                    return x, fx, step * 0.5

        # Main loop
        while not check_time():
            improved = False
            for i in range(dim):
                if check_time():
                    break
                x, fx, steps[i] = line_search_1d(x, i, steps[i], fx)
                # If step became very small, reset it to allow exploration?
                # Or keep it small to exploit.
                # Let's add a small random perturbation if step is tiny to avoid local minima?
                # No, f14 is unimodal.
                
                if steps[i] < 1e-4:
                    steps[i] = 1.0 # Reset to explore again? 
                    # This might be bad if we are close to min.
                    # Let's just let it be small.
            
            # Check for stagnation
            # If no improvement in a full cycle, we might be stuck.
            # But coordinate descent on unimodal functions should converge.

        return x, fx
# EVOLVE-BLOCK-END
