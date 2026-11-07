"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-Evolution Strategy with CMA-like adaptation for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        # Initialize at box center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Step size initialization
        sigma = 2.0

        # Main (1+1)-ES loop
        while calls < budget and time_ok():
            # Generate one mutant
            z = np.random.randn(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls += 1

            if f_trial < fx:
                # Accept: move to trial point
                x = x_trial
                fx = f_trial
                if fx < best_f:
                    best_f = fx
                    best_x = x_trial.copy()
                # Increase step size (success)
                sigma *= 1.5
            else:
                # Reject: stay at current point
                # Decrease step size (failure)
                sigma *= 0.87

            # If step size becomes too small, we can break or continue
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
