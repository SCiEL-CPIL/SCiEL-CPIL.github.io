"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight evolutionary multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return indices of non-dominated points (lightweight version)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        # A point is non-dominated if no other point is better in all objectives
        # and at least as good in all
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Initial population: mix of random and slightly perturbed random
        pop_size = min(30, self.budget // 6)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best known non-dominated points
        all_F = F.copy()

        # Adaptive mutation scale: start large, shrink
        base_scale = 0.3
        min_scale = 0.02

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Select parent from non-dominated front with some randomness
            # Use tournament-like selection: pick 2 from ND front, take better
            if len(nd_indices) >= 2:
                i = nd_indices[rng.integers(len(nd_indices))]
                j = nd_indices[rng.integers(len(nd_indices))]
                # Pick the one with lower sum of objectives (rough fitness)
                if np.sum(F[i]) <= np.sum(F[j]):
                    parent_idx = i
                else:
                    parent_idx = j
            else:
                parent_idx = nd_indices[0]

            # Adaptive mutation: scale decreases as we use more budget
            progress = used / self.budget
            scale = base_scale * (1 - progress) + min_scale * progress

            # Gaussian mutation with adaptive scale
            mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst individual in the population
            # Use sum of objectives as a simple fitness measure
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Also maintain a global archive of non-dominated points
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
