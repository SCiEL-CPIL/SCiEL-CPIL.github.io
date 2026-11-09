"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multiple restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Use multiple restarts with different random seeds
        # Allocate budget across restarts
        n_restarts = 5
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:  # Check deadline with margin
                break

            # Different starting points for each restart
            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = np.ones(dim) * 3.0
            elif restart == 2:
                x = np.ones(dim) * -3.0
            elif restart == 3:
                rng = np.random.default_rng(42 + restart)
                x = rng.uniform(-4, 4, dim)
            else:
                rng = np.random.default_rng(100 + restart)
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive step size
            sigma = 1.5
            success_count = 0
            total_steps = 0

            while total_calls < budget and time.time() < deadline - 5.0:
                rng = np.random.default_rng(12345 + restart * 1000 + total_steps)
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                total_steps += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                    success_count += 1
                else:
                    sigma *= 0.85

                # Adaptive sigma based on success rate (1/5 rule approximation)
                if total_steps % 10 == 0:
                    rate = success_count / total_steps
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    success_count = 0
                    total_steps = 0

                # If we're stuck, restart locally
                if sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
