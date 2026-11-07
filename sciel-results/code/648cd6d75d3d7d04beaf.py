# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using batched island-based search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(12345)

        # --- Phase 1: Initialize with a larger diverse population ---
        n_init = 30
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Track all evaluated solutions
        all_X = [X]
        all_F = [F]

        # Adaptive mutation scale: start large, shrink over time
        base_scale = 0.3
        min_scale = 0.05
        total_planned = budget - n_init
        batch_size = 10

        # --- Phase 2: Iterative batched improvement ---
        n_batches = total_planned // batch_size + 1

        for batch_idx in range(n_batches):
            if time.time() > deadline - 5:
                break
            if used >= budget:
                break

            # Update scale based on progress
            progress = batch_idx / max(1, n_batches - 1)
            scale = base_scale - (base_scale - min_scale) * progress

            # Select parents from current non-dominated set
            # Flatten all stored solutions
            flat_X = np.vstack(all_X)
            flat_F = np.vstack(all_F)
            n_total = len(flat_F)

            # Find non-dominated indices
            nd_mask = self._find_non_dominated(flat_F)
            nd_X = flat_X[nd_mask]
            nd_F = flat_F[nd_mask]
            n_nd = len(nd_X)

            # Generate children
            children_X = []
            children_F = []

            for _ in range(batch_size):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                # Pick a random parent from non-dominated set
                if n_nd > 0:
                    parent_idx = rng.integers(0, n_nd)
                    parent = nd_X[parent_idx]
                else:
                    parent = flat_X[rng.integers(0, n_total)]

                # Gaussian perturbation with adaptive scale
                child = parent + scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if children_X:
                children_X = np.array(children_X)
                children_F = np.array(children_F)
                all_X.append(children_X)
                all_F.append(children_F)

        # --- Final: collect all non-dominated solutions ---
        final_X = np.vstack(all_X)
        final_F = np.vstack(all_F)
        nd_mask = self._find_non_dominated(final_F)
        return final_X[nd_mask]

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points in F (minimization)."""
        n = len(F)
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask

# EVOLVE-BLOCK-END
