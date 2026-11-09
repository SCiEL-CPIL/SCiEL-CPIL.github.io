"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with fast step size adaptation, multi-start exploration, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill

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
        # Lunacek bi-Rastrigin has a global minimum near 0 and a local minimum at offset
        # Try several starting points including origin and perturbed versions
        init_points = [
            np.zeros(n),
            rng.uniform(-2, 2, n),
            rng.uniform(-4, 4, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # Phase 1: Main (1+1)-ES with fast adaptation and restarts
        # Use 1/5 rule for step size: increase on success, decrease on failure
        max_stagnation_per_run = 30
        max_restarts = 8

        for restart in range(max_restarts):
            if calls_made >= total_budget or time.time() >= deadline - 10:
                break

            # Start from best point with perturbation for later restarts
            if restart == 0:
                x = best_x.copy()
            else:
                # Perturb best point to escape local minimum
                perturb_scale = 0.5 + 0.3 * restart
                x = best_x + perturb_scale * rng.standard_normal(n)
                x = np.clip(x, -5, 5)

            fx = safe_func(x)
            if fx >= best_f:
                # If perturbation didn't help, start from best
                x = best_x.copy()
                fx = best_f

            sigma = 1.5 if restart == 0 else 0.8 + 0.2 * restart
            stagnation_count = 0

            while calls_made < total_budget and time.time() < deadline - 5:
                # Generate offspring
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                # 1/5 rule step size adaptation
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stagnation_count = 0
                else:
                    sigma *= 0.8
                    stagnation_count += 1

                # Check for stagnation
                if stagnation_count >= max_stagnation_per_run:
                    break

        # Phase 2: Local refinement with aggressive step size decay
        if time.time() < deadline - 5 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            stagnation_count = 0
            max_stagnation = 15

            while calls_made < total_budget and time.time() < deadline - 5:
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    stagnation_count = 0
                else:
                    sigma *= 0.85
                    stagnation_count += 1

                if stagnation_count >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
