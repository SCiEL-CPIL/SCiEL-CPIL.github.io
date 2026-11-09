"""Multi-start adaptive (1+1)-ES with deadline guard for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        calls_made = 0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start: use diverse initial points to address 101 local minima
        # Budget allocation: spend remaining budget across multiple restarts
        # Each restart gets a fraction of the budget, decreasing as we go

        # First, do a quick exploration with several random starts
        n_initial = 10
        budget_per_initial = max(5, (self.budget - n_initial) // 10)

        # Generate diverse initial points using a grid-like + random approach
        initial_points = []
        # Center
        initial_points.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[i] = sign * 4.0
                initial_points.append(pt)
        # Random points
        while len(initial_points) < n_initial:
            initial_points.append(rng.uniform(-4, 4, self.dim))
        initial_points = np.array(initial_points[:n_initial])

        # Evaluate initial points
        for pt in initial_points:
            if time.time() > deadline:
                break
            evaluate(pt)

        # Now do adaptive (1+1)-ES from the best point, with restarts
        # on stagnation
        sigma = 1.5
        stagnation = 0
        max_stagnation = 50

        # Adaptive sigma with 1/5 rule
        success_window = []
        window_size = 15

        while calls_made < self.budget:
            if time.time() > deadline:
                break

            x = best_x.copy()
            fx = best_f

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = evaluate(y)

            if fy < fx:
                # Success
                x, fx = y, fy
                success_window.append(1)
                # Increase sigma
                sigma *= 1.2
                stagnation = 0
            else:
                success_window.append(0)
                sigma *= 0.9
                stagnation += 1

            # Apply 1/5 success rule
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window)
                if success_rate > 0.4:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.7
                success_window = success_window[-window_size // 2:]

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Restart on stagnation
            if stagnation >= max_stagnation:
                # Try a new random start near current best or fresh random
                if rng.random() < 0.5:
                    # Perturb current best
                    x_new = np.clip(best_x + rng.standard_normal(self.dim) * 2.0, -5, 5)
                else:
                    # Fresh random point
                    x_new = rng.uniform(-4, 4, self.dim)

                fx_new = evaluate(x_new)
                if fx_new < best_f:
                    best_x = x_new.copy()
                    best_f = fx_new

                # Reset stagnation and sigma
                stagnation = 0
                sigma = 1.0
                success_window = []

        return best_x, best_f
# EVOLVE-BLOCK-END
