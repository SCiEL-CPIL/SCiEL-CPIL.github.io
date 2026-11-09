"""Multi-objective optimizer for RE21 using hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume indicator for selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Archive size limit
        max_archive = 100
        initial_pop = 20

        # Initialize
        X_archive = lo + (hi - lo) * rng.random((initial_pop, self.dim))
        F_archive = np.array([problem(x) for x in X_archive], dtype=np.float64)
        used = initial_pop

        # Filter to non-dominated
        X_archive, F_archive = self._prune_archive(X_archive, F_archive, ref_point, max_archive)

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Generate offspring
            n_offspring = 20
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))

            for i in range(n_offspring):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break

                # Mix of exploration and exploitation
                if rng.random() < 0.3:
                    # Exploration: random point
                    child = lo + (hi - lo) * rng.random(self.dim)
                else:
                    # Exploitation: mutate a random point from archive
                    parent_idx = rng.integers(len(X_archive))
                    parent = X_archive[parent_idx]
                    # Gaussian mutation
                    sigma = 0.1 * (hi - lo)
                    child = parent + rng.normal(0, 1, self.dim) * sigma
                    child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine
            combined_X = np.vstack([X_archive, offspring_X])
            combined_F = np.vstack([F_archive, offspring_F])

            # Prune to non-dominated and limit size
            X_archive, F_archive = self._prune_archive(combined_X, combined_F, ref_point, max_archive)

        return X_archive

    def _prune_archive(self, X, F, ref_point, max_size):
        """Prune archive to non-dominated points and limit size by hypervolume contribution."""
        n = len(X)
        if n == 0:
            return X, F

        # 1. Remove dominated points
        non_dom_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dom_mask[i]:
                continue
            for j in range(n):
                if i == j or not non_dom_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dom_mask[i] = False
                    break

        X_nd = X[non_dom_mask]
        F_nd = F[non_dom_mask]

        if len(X_nd) <= max_size:
            return X_nd, F_nd

        # 2. If too many, prune based on hypervolume contribution
        # For 2D, we can compute contribution efficiently
        # Sort by first objective
        sort_idx = np.argsort(F_nd[:, 0])
        X_sorted = X_nd[sort_idx]
        F_sorted = F_nd[sort_idx]

        contributions = np.zeros(len(F_sorted))
        # Hypervolume contribution in 2D:
        # A point (f1, f2) contributes the area of the rectangle
        # [f1, next_f1] x [f2, ref2] if it's on the Pareto front
        # Actually, simpler: sort by f1 ascending.
        # The contribution of point i is (f1[i+1] - f1[i]) * (ref2 - f2[i])
        # But we need to handle the last point and ensure we only count non-dominated parts.
        # Since they are all non-dominated, sorting by f1 ascending implies f2 descending.

        # Check if strictly non-dominated and sorted
        # f1 increasing, f2 decreasing
        for i in range(len(F_sorted) - 1):
            # Width in f1
            w = F_sorted[i+1, 0] - F_sorted[i, 0]
            if w <= 0: w = 0
            # Height in f2
            h = ref_point[1] - F_sorted[i, 1]
            if h <= 0: h = 0
            contributions[i] = w * h

        # Last point
        w = ref_point[0] - F_sorted[-1, 0]
        if w <= 0: w = 0
        h = ref_point[1] - F_sorted[-1, 1]
        if h <= 0: h = 0
        contributions[-1] = w * h

        # Remove points with smallest contributions
        if len(contributions) > max_size:
            # Keep top max_size contributions
            # But we must be careful: removing a point might change the contribution of neighbors.
            # For a quick prune, just removing the smallest is a heuristic.
            # A better way is to iteratively remove the point with the smallest contribution.
            # Given max_size is 100 and n is small, iterative removal is fine.
            keep_indices = np.arange(len(F_sorted))
            current_contributions = contributions.copy()
            current_F = F_sorted.copy()
            current_X = X_sorted.copy()
            
            while len(keep_indices) > max_size:
                # Find index of min contribution
                min_idx = np.argmin(current_contributions)
                # Remove it
                keep_indices = np.delete(keep_indices, min_idx)
                
                # Recalculate contributions for the remaining set
                # This is O(N^2) total which is fine for N=100-200
                F_rem = current_F[keep_indices]
                X_rem = current_X[keep_indices]
                
                # Re-sort and recalc
                sort_idx_rem = np.argsort(F_rem[:, 0])
                F_rem_sorted = F_rem[sort_idx_rem]
                
                new_contribs = np.zeros(len(F_rem_sorted))
                for i in range(len(F_rem_sorted) - 1):
                    w = F_rem_sorted[i+1, 0] - F_rem_sorted[i, 0]
                    h = ref_point[1] - F_rem_sorted[i, 1]
                    new_contribs[i] = max(0, w) * max(0, h)
                w = ref_point[0] - F_rem_sorted[-1, 0]
                h = ref_point[1] - F_rem_sorted[-1, 1]
                new_contribs[-1] = max(0, w) * max(0, h)
                
                current_contributions = new_contribs
                current_F = F_rem_sorted
                current_X = X_rem[sort_idx_rem]
                keep_indices = np.arange(len(F_rem_sorted))
            
            return current_X, current_F

        return X_nd, F_nd
# EVOLVE-BLOCK-END
