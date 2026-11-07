"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: global random sampling, then per-dimension
    adaptive step-size (1+1)-ES local refinement, with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 290.0  # 10s safety margin before 300s wall-clock

        def time_ok():
            return time.time() < DEADLINE

        def safe_call(x):
            try:
                return func(np.clip(x, -5, 5))
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: Global exploration – random sampling
        n_global = min(50, budget // 5)
        best_x = None
        best_f = float('inf')

        for _ in range(n_global):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: Local refinement with per-dimension adaptive step size
        x = best_x
        fx = best_f
        sigma = np.ones(dim) * 1.0  # per-dimension step sizes
        success_count = 0
        total_count = 0
        remaining = budget - n_global

        for _ in range(remaining):
            if not time_ok():
                break

            # Generate a random direction
            direction = rng.standard_normal(dim)
            # Scale by per-dimension sigma
            step = sigma * direction
            y = np.clip(x + step, -5, 5)
            fy = safe_call(y)

            total_count += 1

            if fy < fx:
                # Success: accept and adapt sigmas
                x = y
                fx = fy
                # Increase sigma in directions that contributed
                sigma *= 1.2
                success_count += 1
            else:
                # Failure: decrease sigma
                sigma *= 0.85

            # Periodically rescale sigma to avoid overflow/underflow
            if total_count % 20 == 0:
                # Normalize sigma to have mean ~1
                mean_sigma = np.mean(sigma)
                if mean_sigma > 0:
                    sigma = sigma / mean_sigma * 1.0
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-4, 10.0)

            # 1/5 success rule: if success rate is too high or too low, adjust
            if total_count >= 10:
                rate = success_count / total_count
                if rate > 0.3:
                    sigma *= 1.5
                    success_count = 0
                    total_count = 0
                elif rate < 0.1:
                    sigma *= 0.7
                    success_count = 0
                    total_count = 0

        return x, fx
# EVOLVE-BLOCK-END
