"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-Start (1+1)-ES with stagnation-based restarts for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        t0 = time.time()
        deadline = t0 + 280.0

        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Quasi-random initialization for broad early coverage ---
        # Use a Sobol-like sequence via Halton for better space-filling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_init = 30
        best_x = None
        best_f = np.inf
        for i in range(1, n_init + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = np.array([halton(i, b) for b in range(2, d + 2)])
            x = -5.0 + 10.0 * x
            x = clip(x)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: Multi-Start (1+1)-ES with stagnation restarts ---
        # Strategy: run (1+1)-ES from current best; if no improvement for
        # 'stall_limit' steps, restart from a new point (random or perturbed best).
        stall_limit = 15
        max_sigma = 4.0
        min_sigma = 1e-4
        inc_factor = 1.3
        dec_factor = 0.75

        while calls < budget and time.time() < deadline:
            # Run one (1+1)-ES trajectory from best_x
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            stall = 0

            while calls < budget and time.time() < deadline:
                y = clip(x + sigma * rng.standard_normal(d))
                fy = float(func(y))
                calls += 1

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma = min(sigma * inc_factor, max_sigma)
                    stall = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= dec_factor
                    stall += 1

                if sigma < min_sigma or stall >= stall_limit:
                    break

            # Stagnation detected: restart from a new location
            if calls >= budget or time.time() > deadline:
                break

            # Choose restart strategy:
            # 50%: fully random point, 50%: perturbation of global best
            if rng.random() < 0.5:
                x_new = clip(rng.uniform(-5, 5, d))
            else:
                x_new = clip(best_x + 1.5 * rng.standard_normal(d))

            fx_new = float(func(x_new))
            calls += 1

            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()

            # If we've been stuck for a long time, try a larger jump
            # (implicit via the 50% random restart above)

        return best_x, best_f
# EVOLVE-BLOCK-END
