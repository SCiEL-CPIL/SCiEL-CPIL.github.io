"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        # Start at origin: shift is in [-4,4]^5, so origin is a good prior
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 1.0
        n_calls = 1
        # Phase 1: exploration with larger steps
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            n_calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
        # Phase 2: polish with very small steps if time remains
        if time.time() < deadline - 10:
            sigma_polish = max(sigma * 0.1, 1e-5)
            for _ in range(min(50, self.budget - n_calls)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                n_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 0.9
        return x, fx
# EVOLVE-BLOCK-END
