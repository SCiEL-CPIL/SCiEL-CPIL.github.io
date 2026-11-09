"""Multi-objective optimizer on RE37 – refined evolutionary search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with non-dominated parent selection,
    adaptive mutation, and a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300 s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Population size: use a reasonable chunk of the budget
        pop_size = min(40, max(10, self.budget // 5))
        remaining = self.budget

        # Initial random population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        remaining -= pop_size

        def dominates(a, b):
            """True if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return mask

        # Main evolutionary loop
        while remaining > 0 and (time.time() - t0) < deadline:
            # Check deadline frequently
            if (time.time() - t0) >= deadline:
                break

            # Identify non-dominated parents
            nd_mask = non_dominated_mask(F)
            n_nd = np.sum(nd_mask)
            if n_nd == 0:
                parents_idx = np.arange(len(F))
            else:
                parents_idx = np.where(nd_mask)[0]

            # Select a parent (random from non-dominated)
            p_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[p_idx].copy()

            # Adaptive mutation: scale by distance to bounds and generation progress
            # Use a mix of Gaussian and Lerp mutations
            r = rng.random()
            if r < 0.5:
                # Gaussian mutation with adaptive step
                step = 0.05 + 0.15 * (1.0 - remaining / self.budget)
                child = parent + step * (hi - lo) * rng.standard_normal(self.dim)
            else:
                # Crossover-like: blend with another non-dominated parent
                if len(parents_idx) > 1:
                    other_idx = parents_idx[rng.integers(len(parents_idx))]
                    alpha = rng.uniform(0.2, 0.8)
                    child = alpha * parent + (1.0 - alpha) * X[other_idx]
                else:
                    child = parent + 0.1 * (hi - lo) * rng.standard_normal(self.dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            remaining -= 1

            # Replace the worst point (by sum of objectives) with the child
            # This maintains diversity by always having a replacement slot
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
