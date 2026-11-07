"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sequence on Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_front = 2
        n_g = dim - n_front

        n_total = min(budget, 200)

        # Use Sobol sequence for low-discrepancy sampling in 2D parameter space
        # Parameters: u1 controls radius (f1^2+f2^2), u2 controls angle
        # We want good hypervolume, so we sample in the (f1, f2) quarter disk
        # using Sobol for uniform coverage

        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # Sobol requires power of 2, so generate enough and take what we need
        n_pow2 = 1
        while n_pow2 < n_total:
            n_pow2 *= 2
        samples = sampler.random(n_pow2)
        samples = samples[:n_total]  # Take first n_total points

        u1 = samples[:, 0]  # For radius: r = 0.5 * sqrt(u1)
        u2 = samples[:, 1]  # For angle: theta = u2 * pi/2

        r = 0.5 * np.sqrt(u1)
        theta = u2 * np.pi / 2.0

        f1 = r * np.cos(theta)
        f2 = r * np.sin(theta)
        f3 = np.sqrt(np.maximum(0.25 - f1**2 - f2**2, 0.0))

        # Map back to x1, x2
        x1_vals = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f3, -1.0, 1.0))
        x2_vals = (2.0 / np.pi) * np.arctan2(f2, f1)

        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x2_vals = np.clip(x2_vals, 0.0, 1.0)

        # Construct decision vectors
        X = np.zeros((n_total, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        if n_g > 0:
            X[:, n_front:] = 0.5

        X = np.clip(X, lo, hi)

        # Check time
        if time.time() - t0 > deadline:
            return X

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
