"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global sampling + (1+1)-ES with 1/5 rule and restarts, time-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315.0  # 15s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # ---- Phase 1: global sampling to find the basin of attraction ----
        x = np.zeros(dim)
        fx = np.inf
        calls = 0

        # Latin-hypercube sample for good coverage
        n_lh = min(80, budget)
        for i in range(n_lh):
            if calls >= budget or time.time() >= deadline:
                break
            # stratified sampling per dimension
            xi = np.array([(rng.random() + i) / n_lh for _ in range(dim)])
            x = np.clip(xi, -5, 5)
            fx = func(x)
            calls += 1
            if np.isfinite(fx):
                x_best = x.copy()
                fx_best = fx
                break

        # If the LHS point is finite, we have a starting point.
        # Otherwise, do random sampling until we get a finite value.
        if not np.isfinite(fx):
            for _ in range(min(50, budget - calls)):
                if calls >= budget or time.time() >= deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                if np.isfinite(fx):
                    x_best = x.copy()
                    fx_best = fx
                    break
            if not np.isfinite(fx):
                # All evaluations returned inf? Return best effort.
                return x, fx

        x = x_best.copy()
        fx = fx_best

        # ---- Phase 2: (1+1)-ES with 1/5 success rule and restarts ----
        sigma = 1.5
        success = 0
        window = 0
        no_improve = 0
        best_fx = fx
        best_x = x.copy()

        while calls < budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if np.isfinite(fy) and fy < fx:
                x, fx = y, fy
                success += 1
                no_improve = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # 1/5 rule: increase sigma if success rate > 0.2
                if success / max(1, success + (window - success)) > 0.2:
                    sigma *= 1.2
            else:
                # No improvement; check if we should restart
                no_improve += 1
                if no_improve > 30:
                    # Restart: jump to a new random point, keep best
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    if calls >= budget or time.time() >= deadline:
                        break
                    if np.isfinite(fx) and fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    no_improve = 0
                    sigma = 1.0
                    success = 0
                    window = 0
                    continue
                # 1/5 rule: decrease sigma if success rate <= 0.2
                if success / max(1, success + (window - success)) <= 0.2:
                    sigma *= 0.85

            window += 1
            if window >= 10:
                success = 0
                window = 0

        # Return the best point found
        if np.isfinite(best_fx):
            return best_x, best_fx
        return x, fx
# EVOLVE-BLOCK-END
