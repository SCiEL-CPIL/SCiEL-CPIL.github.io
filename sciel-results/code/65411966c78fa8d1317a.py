"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        pop = int(min(30, self.budget // 5))
        X = lo + range_ * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop

        while used < self.budget:
            if time.time() > deadline:
                break

            # Fast vectorized dominance check: for each point, is it dominated by any other?
            # F shape: (n, n_obj)
            # For point i, dominated if exists j: F[j] <= F[i] and F[j] < F[i] in at least one obj
            # Vectorized: for each pair (j, i), check if j dominates i
            # Use broadcasting: F[j] <= F[i] for all obj, and F[j] < F[i] for at least one obj
            # This is O(n^2 * n_obj) but with numpy it's fast for n=30
            n = len(F)
            # F_j <= F_i for all objectives: shape (n, n, n_obj) -> (n, n)
            le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)  # (j, i): F[j] <= F[i]
            lt = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)   # (j, i): F[j] < F[i] in some obj
            dominates = le & lt  # (j, i): j dominates i
            np.fill_diagonal(dominates, False)
            is_dominated = np.any(dominates, axis=0)  # True if point i is dominated by some j

            if np.any(~is_dominated):
                nd_mask = ~is_dominated
                parents_idx = np.where(nd_mask)[0]
                parent_idx = parents_idx[rng.integers(len(parents_idx))]
            else:
                parent_idx = rng.integers(n)

            parent = X[parent_idx]

            # Adaptive step: start at 0.15, shrink to 0.03
            progress = used / self.budget
            step = 0.15 - 0.12 * progress

            child = parent + step * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace: prefer dominated points, pick worst by sum among them
            if np.any(is_dominated):
                dom_idx = np.where(is_dominated)[0]
                worst_in_dom = dom_idx[np.argmax(np.sum(F[dom_idx], axis=1))]
                X[worst_in_dom] = child
                F[worst_in_dom] = fc
            else:
                # No dominated points; replace the one with highest sum
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
# EVOLVE-BLOCK-END
