"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: detect gradient, jump to corner, then refine with CMA-like steps."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1

        # Phase 1: Estimate gradient direction via forward differences from origin
        h = 4.0
        x0 = np.zeros(dim)
        f0 = fx_best
        grad_signs = np.zeros(dim)

        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            e = np.zeros(dim)
            e[i] = h
            xp = np.clip(x0 + e, -5, 5)
            fp = func(xp)
            calls += 1
            if fp < f0:
                grad_signs[i] = -1.0
            else:
                grad_signs[i] = 1.0

        # Phase 2: Jump to the corner predicted by gradient signs
        x_corner = np.where(grad_signs > 0, -5.0, 5.0)
        if not np.array_equal(x_corner, x_best):
            fx_corner = func(x_corner)
            calls += 1
            if fx_corner < fx_best:
                x_best = x_corner.copy()
                fx_best = fx_corner

        # Phase 3: Local refinement - use a more aggressive step-size adaptation
        # Start with a larger sigma since we're likely near the optimum already
        x = x_best.copy()
        fx = fx_best
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 5.0

        # Use a (1+1)-ES with better step adaptation
        while calls < budget and time_ok():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y.copy(), fy
                if fx < fx_best:
                    x_best, fx_best = x.copy(), fx
                # Increase sigma on success (but cap it)
                sigma = min(sigma * 1.2, sigma_max)
            else:
                # Decrease sigma on failure
                sigma = max(sigma * 0.7, sigma_min)

            # Early termination if sigma is very small and we're converging
            if sigma < 1e-12:
                break

        # Phase 4: Final polish - try very fine steps around the best point
        if calls < budget and time_ok():
            sigma_fine = 1e-5
            max_fine = min(200, budget - calls)
            for _ in range(max_fine):
                if not time_ok():
                    break
                y = np.clip(x_best + sigma_fine * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_best:
                    x_best, fx_best = y.copy(), fy
                    # Increase sigma slightly on success
                    sigma_fine = min(sigma_fine * 1.05, 1e-2)
                else:
                    # Decrease sigma on failure
                    sigma_fine = max(sigma_fine * 0.9, 1e-10)

        return x_best, fx_best
# EVOLVE-BLOCK-END
