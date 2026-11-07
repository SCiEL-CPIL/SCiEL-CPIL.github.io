"""Constrained random search within shrinking hypersphere for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Constrained random search: sample within a shrinking hypersphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # WFG9 Pareto front is near the origin in the decision space.
        # The bounds for WFG9 are typically [-50, 50] per dimension.
        # We sample within a hypersphere centered at the origin.
        # Radius shrinks linearly from an initial moderate value to a very small value.
        # Smaller initial radius ensures distance variables are small (non-dominated).
        
        r_init = 5.0
        r_final = 0.01

        # Track non-dominated points
        F_list = []  # list of objective vectors
        X_list = []  # list of decision vectors

        for i in range(budget):
            # Linearly shrink radius
            t = i / max(1, budget - 1)
            r = r_init + (r_final - r_init) * t

            # Sample a point uniformly within the hypersphere of radius r centered at origin
            # Using global np.random to respect evaluator's seed
            direction = np.random.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.ones(dim) / np.sqrt(dim)
            else:
                direction /= norm
            u = np.random.random()
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
