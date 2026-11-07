"""Optimized solver for BBOB-style Sphere function with hidden shift/rotation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer exploiting the quadratic structure of Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        def check_time():
            return time.time() > deadline

        # Phase 1: Find the minimizer z using central differences
        # Sphere: f(x) = ||R(x - z)||^2 + offset
        # Gradient at x: 2 * R^T R (x - z) = 2 * (x - z)  (since R is orthogonal)
        # So gradient = 2*(x - z), meaning z = x - gradient/2

        # We'll estimate z from a single point using central differences
        # f(x + h*e_i) - f(x - h*e_i) = 2h * (R^T R e_i) dot (x - z)
        # Since R^T R = I, this is 2h * (x_i - z_i)
        # So z_i = x_i - (f(x+h*e_i) - f(x-h*e_i)) / (2h)

        # Choose a good starting point and step size
        x0 = np.zeros(dim)
        h = 0.01  # small step for accurate gradient estimate

        # Evaluate f at x0
        fx0 = func(x0)
        calls = 1

        # Estimate z using central differences
        z = np.zeros(dim)
        for i in range(dim):
            if check_time():
                break
            x_plus = x0.copy()
            x_plus[i] += h
            x_minus = x0.copy()
            x_minus[i] -= h
            fp = func(x_plus)
            calls += 1
            fm = func(x_minus)
            calls += 1
            # Gradient component: (fp - fm) / (2h) = 2 * (x0_i - z_i)
            # So z_i = x0_i - (fp - fm) / (4h)
            z[i] = x0[i] - (fp - fm) / (4.0 * h)

        # Clip z to the box
        z = np.clip(z, -5, 5)

        # Evaluate at z
        fz = func(z)
        calls += 1
        best_x = z.copy()
        best_fx = fz

        # If we're already very close, we can stop early
        if check_time() or calls >= budget:
            return best_x, best_fx

        # Phase 2: Refine with local optimization if needed
        # Use a simple iterative approach: evaluate nearby points to verify
        # and potentially improve

        # Try a few points around z to see if we can do better
        rng = np.random.default_rng(42)
        remaining = budget - calls

        # If we have budget left, do a local search around z
        if remaining > 10:
            # Try points on a small grid around z
            for trial in range(min(remaining - 1, 50)):
                if check_time():
                    break
                # Random perturbation around z
                scale = 0.1 * (0.5 ** (trial // 10))
                x_trial = z + scale * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                if f_trial < best_fx:
                    best_x = x_trial.copy()
                    best_fx = f_trial

        return best_x, best_fx
# EVOLVE-BLOCK-END
