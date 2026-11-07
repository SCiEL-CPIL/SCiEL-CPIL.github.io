"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Initial point
        x = np.zeros(self.dim)
        fx = func(x)
        call_count = 1

        # Adaptive step size
        sigma = 1.0
        sigma_min = 1e-4
        sigma_max = 5.0
        success_window = 10
        successes = 0

        # Track best
        best_x = x.copy()
        best_f = fx

        # CMA-ES-like covariance matrix (diagonal start)
        # Use a simple adaptive covariance: diagonal with per-dimension scales
        diag_sigma = np.ones(self.dim) * sigma

        while call_count < self.budget:
            # Time guard
            if time.time() > deadline:
                break

            # Generate candidate
            step = diag_sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            call_count += 1

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Accept/reject
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                # Increase step size
                diag_sigma *= 1.1
                diag_sigma = np.clip(diag_sigma, sigma_min, sigma_max)
            else:
                # Decrease step size
                diag_sigma *= 0.9
                diag_sigma = np.clip(diag_sigma, sigma_min, sigma_max)

            # Periodically reset success counter
            if successes >= success_window:
                successes = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
