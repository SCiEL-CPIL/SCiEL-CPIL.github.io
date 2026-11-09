"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        dim = self.dim
        total_budget = self.budget

        # Split budget across restarts
        num_restarts = 8
        budget_per_restart = max(30, total_budget // num_restarts)

        best_x = None
        best_fx = np.inf

        for restart in range(num_restarts):
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Different initialization per restart
            rng = np.random.default_rng(42 + restart * 1000)

            # Mix of strategies: some near origin, some random
            if restart == 0:
                x = np.zeros(dim)
            elif restart < 4:
                # Moderate exploration
                x = rng.uniform(-2, 2, dim)
            else:
                # Wider exploration
                x = rng.uniform(-4.5, 4.5, dim)

            fx = func(x)
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

            sigma = 2.5
            calls_in_restart = 1

            # Adaptive sigma with 1/5 success rule
            window_size = 10
            window = []

            for _ in range(budget_per_restart - 1):
                # Check deadline
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                else:
                    window.append(0)

                # Update best
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()

                # Adapt sigma
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.25:
                        sigma *= 1.25
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    window = window[-window_size:]
                else:
                    if len(window) > 5:
                        recent_success = sum(window) / len(window)
                        if recent_success > 0.3:
                            sigma *= 1.1
                        elif recent_success < 0.1:
                            sigma *= 0.85

                # Keep sigma in reasonable range
                sigma = np.clip(sigma, 0.1, 5.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
