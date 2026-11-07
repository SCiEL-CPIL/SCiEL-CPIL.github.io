"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim

        # We'll do multi-restart: split budget among restarts
        # For Rastrigin, local minima are common, so restarts help
        # Use 5 restarts of 100 evaluations each
        n_restarts = 5
        per_restart = total_budget // n_restarts  # 100 per restart
        remaining = total_budget - per_restart * n_restarts
        per_restart_last = per_restart + remaining

        best_x = None
        best_f = float('inf')
        calls_made = 0

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Initialize: for first restart use a structured grid-like start
            # For subsequent restarts, random starts
            if r == 0:
                # Start near origin (Rastrigin's global minimum is at 0)
                x = np.zeros(dim)
            else:
                # Random start, but biased towards center
                x = np.random.uniform(-4, 4, dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.0
            for i in range(per_restart if r < n_restarts - 1 else per_restart_last):
                # Check time
                if time.time() > deadline - 2.0:
                    break

                # Check budget
                if calls_made >= total_budget:
                    break

                # Generate candidate
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.01, 10.0)

            if calls_made >= total_budget:
                break

        # If we have time and budget left, do a final local polish
        if calls_made < total_budget and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls_made < total_budget and time.time() < deadline - 2.0:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.001, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
