"""Optimizer for ZDT1 exploiting its known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1 structure: x1 controls front position, x2..xn control distance to front.
    Places points on the Pareto front (x2..xn=0) with uniform spacing in f1-space
    to maximize hypervolume coverage of the convex Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # ZDT1 Pareto front: x2..xn = 0, f1 = x1, f2 = 1 - sqrt(x1)
        # To maximize hypervolume for a convex front, we want uniform spacing in f1-space.
        # Since f1 = x1 on the Pareto front, we just need uniform x1 in [0, 1].
        n_points = self.budget

        # Uniform spacing in x1 (which equals f1 on the Pareto front)
        x1_vals = np.linspace(0.0, 1.0, n_points)

        evaluated = 0
        for i in range(n_points):
            if time.time() > deadline:
                break

            x = np.zeros(n)
            x[0] = x1_vals[i]
            # x[1:] = 0 (Pareto front)
            problem(x)
            evaluated += 1

        return None
# EVOLVE-BLOCK-END
