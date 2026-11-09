"""Combined optimizer: random-seed + (1+1)-ES with multi-scale restarts and fine polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-scale restarts, adaptive sigma, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        calls_made = 0

        # Phase 1: Random sampling for initial diversity
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

        # Phase 2: Multi-scale (1+1)-ES with restarts
        # Use different sigma scales for different restarts to explore different scales
        sigma_scales = [3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        max_restarts = len(sigma_scales) + 6
        restart_count = 0

        while calls_made < budget and restart_count < max_restarts:
            if time.time() > deadline:
                break

            # Choose starting point for this restart
            if restart_count == 0:
                x = best_x.copy()
                fx = best_f
            else:
                # Mix strategies: perturb best, random, or intermediate
                r = rng.random()
                if r < 0.35:
                    # Perturb best with moderate step
                    x = best_x + 1.5 * rng.standard_normal(dim)
                elif r < 0.65:
                    # Random restart
                    x = rng.uniform(-5, 5, dim)
                else:
                    # Use a point near the best with random offset
                    offset = 2.5 * (rng.random(dim) - 0.5)
                    x = best_x + offset
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                update_best(x, fx)

            # Determine sigma based on restart count
            if restart_count < len(sigma_scales):
                sigma = sigma_scales[restart_count]
            else:
                sigma = 0.01 * (1.0 + rng.random())

            no_improve = 0
            max_no_improve = 20 + 5 * min(restart_count, 3)

            while calls_made < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fy)
                    no_improve = 0
                    # Increase sigma on success but cap it
                    sigma = min(sigma * 1.3, 5.0)
                else:
                    no_improve += 1
                    sigma *= 0.8
                    if no_improve > max_no_improve:
                        break

            restart_count += 1

        # Phase 3: Fine local refinement around best point
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fy)
                    no_improve = 0
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    no_improve += 1
                    sigma *= 0.75
                    if no_improve > 15 or sigma < 0.001:
                        break

        # Phase 4: Final very fine polish if budget remains
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fy)
                else:
                    sigma *= 0.7
                    if sigma < 0.0001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
