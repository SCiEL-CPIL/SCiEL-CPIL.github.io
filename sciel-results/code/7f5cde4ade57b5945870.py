"""Constrained random search within shrinking hypersphere for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Constrained random search: sample within a shrinking hypersphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # WFG9 Pareto front is extremely close to the origin.
        # Use a quadratic shrink to spend more evaluations near the origin.
        r_init = 5.0
        beta = 2.0  # exponent for (1-t)^beta shrink

        F_list = []
        X_list = []

        for i in range(budget):
            t = i / max(1, budget)
            # Quadratic shrink: r = r_init * (1 - t)^beta
            r = r_init * ((1.0 - t) ** beta)

            # Sample uniformly within hypersphere of radius r centered at origin
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.ones(dim) / np.sqrt(dim)
            else:
                direction /= norm
            u = rng.random()
            radius = r * (u ** (1.0 / dim))
            x = direction * radius

            # Clip to bounds
            x = np.clip(x, lo, hi)

            # Evaluate
            f = np.array(problem(x), dtype=float)

            # Store
            F_list.append(f)
            X_list.append(x)

        return X_list
# EVOLVE-BLOCK-END
