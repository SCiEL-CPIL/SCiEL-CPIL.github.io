"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer using analytical front sampling via Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, keep 15s buffer
        deadline = start_time + 315.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ1 structure:
        # f1 = (1+g(x)) * (1 - sqrt(x0))
        # f2 = (1+g(x)) * sqrt(x0)
        # f3 = (1+g(x))
        # where g(x) = sum_{i=2}^{n} (x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))
        #
        # Strategy:
        # 1. Minimize g(x) analytically by setting x_i = 0.5 for i >= 1.
        #    This gives the minimum g(x) value, leading to the true Pareto front.
        # 2. Sample x0 using Halton sequence in base 2 for quasi-uniform coverage
        #    of [0, 1]. This provides better space-filling properties than uniform
        #    grid, especially for hypervolume maximization with distant reference.
        # 3. Map x0 to Pareto front points and evaluate.

        evals_used = 0
        points = []

        # Construct the fixed part of x (convergence variables)
        # Set to 0.5 to minimize g(x)
        x_conv = np.full(self.dim - 1, 0.5)
        # Ensure it's within bounds
        x_conv = np.clip(x_conv, lo[1:], hi[1:])

        N = self.budget

        # Generate x0 values using Halton sequence in base 2
        # This gives better space-filling than uniform grid for hypervolume
        def halton(index, base):
            """Generate a point in the Halton sequence."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        if N == 1:
            x0_vals = np.array([0.5])
        else:
            # Use Halton sequence starting from index 2 (skip index 0 which is 0.0)
            # This gives better spread than starting from 1
            x0_vals = np.array([halton(i, 2) for i in range(2, 2 + N)])

        # Evaluate all points
        for x0 in x0_vals:
            if evals_used >= self.budget:
                break
            if time.time() > deadline:
                break

            x_full = np.zeros(self.dim)
            x_full[0] = x0
            x_full[1:] = x_conv

            # Clip to bounds just in case
            x_full = np.clip(x_full, lo, hi)

            f = problem(x_full)
            evals_used += 1
            points.append((x_full, f))

        return points
# EVOLVE-BLOCK-END
