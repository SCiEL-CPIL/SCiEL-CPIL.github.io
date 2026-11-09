"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a simple evolutionary strategy with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = min(20, self.budget // 10)
        pop_size = max(10, pop_size)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DTLZ4 specific: the first 7 variables control the shape, last 5 are position variables
        # For DTLZ4, position variables (last n_obj-1) should be near 0 or 1 for good solutions
        # Actually for DTLZ4 with 3 objectives, position vars are x[7:] (5 vars)
        # Shape vars are x[:7]

        def is_non_dominated(F):
            """Return boolean array for non-dominated points."""
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

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = is_non_dominated(F)
            if np.any(nd_mask):
                parents = X[nd_mask]
            else:
                parents = X

            # Select a parent
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]

            # Mutation: perturb position variables more aggressively toward 0 or 1
            # For DTLZ4, position variables (last 5) should be extreme
            child = parent.copy()
            for i in range(self.dim):
                if i >= self.dim - (self.n_obj - 1):
                    # Position variable: push toward 0 or 1
                    if rng.random() < 0.5:
                        child[i] = 0.0
                    else:
                        child[i] = 1.0
                else:
                    # Shape variable: small perturbation
                    child[i] += 0.1 * range_[i] * rng.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (highest sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx], F[worst_idx] = child, fc

        return X
# EVOLVE-BLOCK-END
