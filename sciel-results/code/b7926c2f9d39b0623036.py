"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Two-phase optimizer for Bent Cigar:
    Phase 1: Deterministic Sobol sampling for coarse exploration.
    Phase 2: Adaptive (1+1)-ES with 1/5 success rule for fine-tuning.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Sobol sampling for systematic coarse search
        # Use a Sobol sequence to cover the box efficiently
        n_sobol = min(50, budget // 10)  # 50 points or 10% of budget
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)  # in [0, 1]^d
        # Map to [-5, 5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        best_x = None
        best_fx = np.inf

        for i in range(n_sobol):
            if calls_used >= budget or time.time() >= deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Adaptive (1+1)-ES from the best Sobol point
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0  # initial step size
        successes = 0
        total = 0
        success_threshold = 0.2  # 1/5 rule

        rng = np.random.default_rng(123)

        while calls_used < budget and time.time() < deadline:
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                successes += 1
            total += 1

            # Adapt step size based on success rate
            if total >= 5:
                rate = successes / total
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                successes = 0
                total = 0

            # Clamp sigma
            if sigma < 1e-8:
                sigma = 1e-8
            if sigma > 10.0:
                sigma = 10.0

            # Early termination if sigma is very small and we've made good progress
            if sigma < 1e-6 and calls_used > 100:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
