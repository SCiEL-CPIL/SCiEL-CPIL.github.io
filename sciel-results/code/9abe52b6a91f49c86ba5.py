"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with deadline guard and NSGA-II-style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        init_size = min(20, self.budget // 5)
        X = lo + span * rng.random((init_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = init_size

        def is_non_dominated(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        # Main search loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = is_non_dominated(F)
            n_nd = np.sum(nd_mask)

            # Adaptive step size: starts large, shrinks over budget
            progress = used / max(1, self.budget)
            base_step = 0.3 * (1.0 - 0.7 * progress)

            # Select parent: prefer non-dominated, use random among them
            if n_nd > 0:
                nd_indices = np.where(nd_mask)[0]
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
            else:
                parent_idx = rng.integers(len(F))

            # Mutation
            step = base_step * span
            child = X[parent_idx] + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace worst point (simplest: replace point with highest sum of objectives)
            # Better: replace a dominated point if exists
            if n_nd < len(F):
                # Find a dominated point to replace
                dom_mask = ~nd_mask
                dom_indices = np.where(dom_mask)[0]
                # Among dominated, replace the one with highest sum
                replace_idx = dom_indices[np.argmax(F[dom_indices].sum(axis=1))]
            else:
                # All non-dominated: replace the one farthest from ideal in normalized space
                replace_idx = int(np.argmax(F.sum(axis=1)))

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
