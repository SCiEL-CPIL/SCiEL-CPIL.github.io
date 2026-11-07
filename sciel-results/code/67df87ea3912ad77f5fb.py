"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with success-rule step-size adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        sigma = 2.0
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Success rule parameters
        chi_n = np.sqrt(d) * (1 + 0.5 * (1 - 1 / (4 * d)))
        c_sigma = (1 + 2 * min(1 / (d + 2), 1 / (2 * chi_n))) / (d + 4)
        # Actually let's use simpler constants
        c_sigma = 1.0 / (d + 4)
        h_sigma = 1 - np.exp(-0.5 / (d + 4))

        # State for success rule
        s = 0.0

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            # Update best
            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy

            # Success rule step size adaptation
            if fy < fx:
                s = (1 - c_sigma) * s + np.sqrt(d)
                x, fx = y, fy
            else:
                s = (1 - c_sigma) * s

            sigma *= np.exp(c_sigma * (s / h_sigma - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

        return best_x, best_fx


# Helper for exp if needed (actually numpy handles it)
import math
def _exp(x):
    return math.exp(x)

# We need to fix the sigma update to use proper math
# Let me rewrite more carefully

class Optimizer:
    """(1+1)-ES with success-rule step-size adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        sigma = 2.0
        best_x = x.copy()
        best_fx = fx

        # Success rule parameters
        c_sigma = 1.0 / (d + 4)
        h_sigma = 1.0 - math.exp(-0.5 / (d + 4))
        s = 0.0

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)

            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy

            if fy < fx:
                s = (1.0 - c_sigma) * s + math.sqrt(d)
                x, fx = y, fy
            else:
                s = (1.0 - c_sigma) * s

            sigma *= math.exp(c_sigma * (s / h_sigma - 1.0))
            sigma = max(1e-8, min(sigma, 50.0))

        return best_x, best_fx
# EVOLVE-BLOCK-END
