"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def check_deadline():
            return (time.time() - start_time) > deadline

        # Multi-restart strategy
        # Start with 5 restarts, each using a fraction of the budget
        # This allows exploration of different regions of the search space
        n_restarts = 5
        budget_per_restart = budget // n_restarts

        # For each restart, use (1+1)-ES with adaptive step size
        for restart in range(n_restarts):
            if calls_made >= budget or check_deadline():
                break

            # Initialize at a random point in the search space
            # Use different starting points for each restart
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = np.random.uniform(-4, 4, dim)

            # Initial evaluation
            f = call_func(x)
            if calls_made >= budget or check_deadline():
                break

            # Adaptive step size
            sigma = 2.0  # Start with a larger step for exploration
            step_size = 0.5

            # Run (1+1)-ES for this restart
            while calls_made < budget - budget_per_restart and not check_deadline():
                # Generate a single offspring
                z = np.random.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)

                f_new = call_func(x_new)
                if calls_made >= budget or check_deadline():
                    break

                # Update
                if f_new < f:
                    # Success: increase step size
                    x = x_new
                    f = f_new
                    sigma *= 1.2  # Increase step size
                else:
                    # Failure: decrease step size
                    sigma *= 0.8  # Decrease step size

                # Adaptive step size adjustment
                if sigma < 0.01:
                    sigma = 0.01
                elif sigma > 10:
                    sigma = 10

        return best_x, best_f

# EVOLVE-BLOCK-END
