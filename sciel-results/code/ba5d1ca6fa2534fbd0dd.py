"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Random sampling to find a good starting point
        n_initial = min(10, budget)
        for i in range(n_initial):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Multiple restarts with (1+1)-ES
        remaining_budget = budget - calls_made
        n_restarts = min(4, max(1, remaining_budget // 60))
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            # Start from current best with some perturbation
            if restart == 0:
                x = best_x.copy()
            else:
                # Larger perturbation for later restarts to explore more
                scale = 1.0 + 0.5 * restart
                x = best_x + scale * rng.standard_normal(d)
                x = np.clip(x, -5, 5)

            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 0.5
            # Adaptive step size with success rate control
            successes = 0
            total_steps = 0
            window_size = 10

            for step in range(budget_per_restart):
                if calls_made >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = call_func(y)

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    pass

                total_steps += 1

                # Adapt sigma every window_size steps based on success rate
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    # Target success rate ~0.33 for (1+1)-ES
                    if success_rate > 0.4:
                        sigma *= 1.25
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0

                    # Clamp sigma
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
