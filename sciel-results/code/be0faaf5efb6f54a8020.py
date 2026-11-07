"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase hybrid: deterministic radial sweep for Discus axis, then aggressive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Phase 1: Dense deterministic radial sweep from origin to find the Discus axis.
        # The Discus function has extreme anisotropy: a narrow "needle" along one axis.
        # We sample points on a grid along each axis and combinations to locate the minimum direction.
        best_x = np.zeros(dim)
        best_f = np.inf

        # Sample along each axis at several radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        for i in range(dim):
            for r in radii:
                for sign in [1.0, -1.0]:
                    x = np.zeros(dim)
                    x[i] = sign * r
                    x = clip(x)
                    f = func(x)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

        # Also sample some off-axis points to detect the shift
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    for r1 in [1.0, 2.0, 3.0]:
                        for r2 in [0.5, 1.0, 1.5]:
                            x = np.zeros(dim)
                            x[i] = r1
                            x[j] = r2
                            x = clip(x)
                            f = func(x)
                            calls += 1
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()

        # Phase 2: Aggressive (1+1)-ES with adaptive step size
        # Use the best point from Phase 1 as the starting point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        rng = np.random.default_rng(42)

        while calls < budget and time.time() < deadline:
            # Generate mutation
            y = clip(x + sigma * rng.standard_normal(dim))
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3  # Increase step on success
            else:
                sigma *= 0.85  # Decrease step on failure

            # Safety check
            if time.time() > deadline:
                break

        return x, fx
# EVOLVE-BLOCK-END
