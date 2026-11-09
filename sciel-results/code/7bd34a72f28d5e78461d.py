# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and limited random restarts for f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 2.0

        # Start with a large sigma to quickly explore the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1
        sigma = 4.0  # Large initial step for global exploration

        # (1+1)-ES with 1/5 success rule
        # We'll run this for most of the budget, with occasional restarts
        # if we detect stagnation

        stagnation_count = 0
        last_improvement_f = fx
        restart_threshold = 50  # Restart if no improvement for 50 iterations

        for i in range(budget - 1):
            if not time_ok() or calls_made >= budget:
                break

            # Generate mutant
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    stagnation_count = 0
                    # Increase sigma on success (1/5 rule: multiply by ~1.5)
                    sigma *= 1.5
            else:
                # Decrease sigma on failure
                sigma *= 0.87
                stagnation_count += 1

            # Check for stagnation and consider restart
            if stagnation_count >= restart_threshold and calls_made < budget - 50:
                # Try a random restart from a new location
                if rng.random() < 0.3:  # 30% chance to restart
                    x_new = rng.uniform(-4, 4, dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = 2.0  # Reset sigma for new exploration
                        stagnation_count = 0
                    else:
                        # Keep current best, but reduce sigma further
                        sigma *= 0.5
                        stagnation_count = 0

            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
