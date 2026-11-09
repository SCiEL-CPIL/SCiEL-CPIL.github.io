"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with 1/5 success rule step-size adaptation and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 0: Multi-start initialization to find both basins of Lunacek
        init_points = [
            np.zeros(n),
            rng.uniform(-1, 1, n),
            rng.uniform(-2, 2, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
            rng.uniform(-4, 4, n),
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # Phase 1: Coordinate Descent with 1/5 success rule and restarts
        # The 1/5 success rule: if more than 1/5 of recent trials succeed, increase step;
        # if fewer, decrease. This provides principled adaptation.
        
        max_restarts = 8
        max_stagnation_per_run = 40
        window_size = 25
        success_threshold = 1.0 / 5.0

        for restart in range(max_restarts):
            if calls_made >= total_budget or time.time() >= deadline - 10:
                break

            if restart == 0:
                x = best_x.copy()
            else:
                # Perturb best point to escape local minimum
                perturb_scale = 0.3 + 0.2 * restart
                x = best_x + perturb_scale * rng.standard_normal(n)
                x = np.clip(x, -5, 5)

            fx = safe_func(x)
            if fx >= best_f:
                x = best_x.copy()
                fx = best_f

            # Per-coordinate step sizes
            sigma = np.ones(n) * (0.5 if restart == 0 else 0.4 + 0.1 * restart)
            window = []
            stagnation_count = 0
            coord_idx = 0

            while calls_made < total_budget and time.time() < deadline - 5:
                # Try moving along coordinate coord_idx
                d = rng.choice([-1, 1])
                y = x.copy()
                y[coord_idx] += d * sigma[coord_idx]
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    stagnation_count = 0
                else:
                    window.append(0)
                    stagnation_count += 1

                # 1/5 success rule for step size adaptation
                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > success_threshold:
                        # Increase step size for this coordinate
                        sigma[coord_idx] *= 1.2
                    elif success_rate < 0.1:
                        # Decrease step size for this coordinate
                        sigma[coord_idx] *= 0.8
                    window = window[1:]

                # Advance to next coordinate cyclically
                coord_idx = (coord_idx + 1) % n

                # Check for stagnation
                if stagnation_count >= max_stagnation_per_run:
                    break

        # Phase 2: Fine-grained coordinate descent refinement
        if time.time() < deadline - 5 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = np.ones(n) * 0.1
            window = []
            window_size = 15
            stagnation_count = 0
            coord_idx = 0

            while calls_made < total_budget and time.time() < deadline - 5:
                d = rng.choice([-1, 1])
                y = x.copy()
                y[coord_idx] += d * sigma[coord_idx]
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    stagnation_count = 0
                else:
                    window.append(0)
                    stagnation_count += 1

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.2:
                        sigma[coord_idx] *= 1.1
                    elif success_rate < 0.05:
                        sigma[coord_idx] *= 0.9
                    window = window[1:]

                coord_idx = (coord_idx + 1) % n

                if stagnation_count >= 30:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
