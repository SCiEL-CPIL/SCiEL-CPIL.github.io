"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            call_count += 1
            return float(func(x))

        # Phase 1: Multiple restarts with (1+1)-ES
        # Use a portion of budget for exploration, then polish
        n_restarts = 5
        budget_per_restart = max(1, (budget - 1) // n_restarts)

        for r in range(n_restarts):
            if call_count >= budget - 10:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart at origin, others random
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 3.0 / (r + 1)  # decreasing step size for later restarts
            success_count = 0
            total_trials = 0

            for i in range(budget_per_restart):
                if call_count >= budget - 10:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                total_trials += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.25
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adaptive sigma based on success rate (1/5 rule approximation)
                if total_trials >= 10:
                    rate = success_count / total_trials
                    if rate > 0.3:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                    total_trials = 0

        # Phase 2: Local polish around best point with small steps
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for i in range(min(50, budget - call_count - 5)):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.95
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

        # Phase 3: Very fine polish
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for i in range(min(30, budget - call_count - 5)):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.9
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.7

        if best_x is None:
            # Fallback: should not happen
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
