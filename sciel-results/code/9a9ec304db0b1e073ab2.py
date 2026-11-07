# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using batched island-based search with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(12345)

        # --- Phase 1: Smart initialization ---
        # Initialize with points spread across the unit hypercube
        # plus some points biased toward known good regions for RE34
        n_init = 20
        X = lo + (hi - lo) * rng.random((n_init, dim))
        # Add a few points near the center and corners
        center = (lo + hi) / 2.0
        X = np.vstack([X, np.tile(center, (3, 1))])
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = len(X)

        # Maintain a single list of all evaluated solutions
        all_X = [X]
        all_F = [F]

        # --- Helper to get non-dominated set efficiently ---
        def get_nd_set(X_arr, F_arr):
            n = len(F_arr)
            if n <= 1:
                return np.ones(n, dtype=bool)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                # Check if any other point dominates i
                dominated = False
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if dominated:
                    nd[i] = False
            return nd

        # --- Phase 2: Main search loop with adaptive scaling ---
        base_scale = 0.25
        min_scale = 0.02
        batch_size = 8
        total_remaining = budget - used
        n_batches = total_remaining // batch_size + 1

        for batch_idx in range(n_batches):
            if time.time() > deadline - 10:
                break
            if used >= budget:
                break

            # Adaptive scale: larger early, smaller later
            progress = min(1.0, batch_idx / max(1, n_batches * 0.7))
            scale = base_scale - (base_scale - min_scale) * progress

            # Get current non-dominated set
            flat_X = np.vstack(all_X)
            flat_F = np.vstack(all_F)
            n_total = len(flat_F)
            nd_mask = get_nd_set(flat_X, flat_F)
            nd_X = flat_X[nd_mask]
            n_nd = len(nd_X)

            # Generate children
            children_X = []
            children_F = []

            for _ in range(batch_size):
                if used >= budget:
                    break
                if time.time() > deadline - 3:
                    break

                # Select parent: prefer non-dominated, with some exploration
                if n_nd > 0 and rng.random() < 0.8:
                    parent = nd_X[rng.integers(0, n_nd)]
                elif n_total > 0:
                    parent = flat_X[rng.integers(0, n_total)]
                else:
                    parent = (lo + hi) / 2.0

                # Gaussian perturbation
                child = parent + scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if children_X:
                all_X.append(np.array(children_X))
                all_F.append(np.array(children_F))

        # --- Phase 3: Local refinement of best solutions ---
        if time.time() < deadline - 5 and used < budget:
            flat_X = np.vstack(all_X)
            flat_F = np.vstack(all_F)
            nd_mask = get_nd_set(flat_X, flat_F)
            nd_X = flat_X[nd_mask]
            nd_F = flat_F[nd_mask]
            n_nd = len(nd_X)

            # Refine top non-dominated solutions with smaller steps
            n_refine = min(n_nd, 10)
            if n_refine > 0:
                # Sort by sum of objectives to get "best" ones
                sums = nd_F.sum(axis=1)
                best_idx = np.argsort(sums)[:n_refine]

                refine_scale = 0.03
                for idx in best_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline - 2:
                        break

                    parent = nd_X[idx]
                    # Try several small perturbations
                    for _ in range(3):
                        if used >= budget:
                            break
                        if time.time() > deadline - 1:
                            break
                        child = parent + refine_scale * (hi - lo) * rng.standard_normal(dim)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        all_X.append(child.reshape(1, -1))
                        all_F.append(fc.reshape(1, -1))

        # --- Final: collect all non-dominated solutions ---
        final_X = np.vstack(all_X)
        final_F = np.vstack(all_F)
        nd_mask = get_nd_set(final_X, final_F)
        return final_X[nd_mask]

# EVOLVE-BLOCK-END
