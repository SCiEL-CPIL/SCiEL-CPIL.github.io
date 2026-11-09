"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Greedy Non-Dominated Set Expansion."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initialize with a small random set
        init_size = 10
        X = lo + span * np.random.random((init_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = init_size

        # Store non-dominated points as list of (x, f) tuples
        nd_points = []

        def is_dominated(f_new, nd_list):
            """Check if f_new is dominated by any point in nd_list."""
            for f_old in nd_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def dominates(f_new, nd_list):
            """Check if f_new dominates any points in nd_list, return indices to remove."""
            to_remove = []
            for i, f_old in enumerate(nd_list):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    to_remove.append(i)
            return to_remove

        # Initialize nd_points
        for i in range(init_size):
            if not is_dominated(F[i], [f for x, f in nd_points]):
                rem = dominates(F[i], [f for x, f in nd_points])
                for idx in sorted(rem, reverse=True):
                    nd_points.pop(idx)
                nd_points.append((X[i].copy(), F[i].copy()))

        # Main loop
        while evals < self.budget:
            # Decide whether to explore (random) or exploit (perturb)
            if np.random.random() < 0.1:
                # Explore: new random point
                x_new = lo + span * np.random.random(self.dim)
            else:
                # Exploit: perturb a random non-dominated point
                if not nd_points:
                    x_new = lo + span * np.random.random(self.dim)
                else:
                    parent_idx = np.random.randint(len(nd_points))
                    x_parent, f_parent = nd_points[parent_idx]
                    # Perturb with Gaussian noise
                    x_new = x_parent + 0.1 * span * np.random.standard_normal(self.dim)
                    x_new = np.clip(x_new, lo, hi)

            f_new = problem(x_new)
            evals += 1

            # Update non-dominated set
            if not is_dominated(f_new, [f for x, f in nd_points]):
                rem = dominates(f_new, [f for x, f in nd_points])
                for idx in sorted(rem, reverse=True):
                    nd_points.pop(idx)
                nd_points.append((x_new.copy(), f_new.copy()))

        return nd_points
# EVOLVE-BLOCK-END
