"""Rewritten optimizer for BBOB-style suite."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy continuous optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best point and value
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # --- Phase 1: Random sampling for initial diversity ---
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            update_best(x, f)
            if time.time() > deadline:
                break

        # --- Phase 2: Multiple (1+1)-ES restarts with adaptive sigma ---
        # Use remaining budget for restarts
        calls_made = n_init
        restarts = 0
        max_restarts = 15

        while calls_made < budget and restarts < max_restarts:
            if time.time() > deadline:
                break

            # Choose restart point: mix of best-so-far perturbation and random
            if restarts == 0:
                x = best_x.copy()
            elif rng.random() < 0.5:
                # Perturb best
                x = best_x + 2.0 * rng.standard_normal(dim)
            else:
                # Random restart
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            sigma = 1.0
            no_improve = 0
            max_no_improve = 30

            while calls_made < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    no_improve = 0
                    sigma = min(sigma * 1.3, 10.0)
                else:
                    no_improve += 1
                    sigma *= 0.85
                    if no_improve > max_no_improve:
                        break

            restarts += 1

        # --- Phase 3: Local refinement around best point ---
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    sigma *= 0.8
                    if sigma < 0.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
