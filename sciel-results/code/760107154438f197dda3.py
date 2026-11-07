"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase (1+1)-ES: aggressive exploration then polished convergence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        start_time = time.time()
        deadline = start_time + 325.0

        # Phase 1: Aggressive exploration
        sigma = 2.0
        sigma_cap = 15.0
        growth = 1.25
        decay = 0.85
        phase1_end = 300  # switch to polish after ~300 calls or when time suggests

        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Phase transition: switch to polish mode
            if i >= phase1_end:
                # Polish phase
                if sigma > 0.05:
                    sigma = 0.05
                # Geometric decay with adaptive boost
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1  # small boost on success
                else:
                    sigma *= 0.95  # geometric decay
                continue

            # Exploration phase
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * growth, sigma_cap)
            else:
                sigma = max(sigma * decay, 0.1)

        return x, fx
# EVOLVE-BLOCK-END
