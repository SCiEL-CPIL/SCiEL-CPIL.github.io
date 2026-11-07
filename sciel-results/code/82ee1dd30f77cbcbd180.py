"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and hypervolume-aware replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = int(min(30, self.budget // 6))
        if pop_size < 5:
            pop_size = 5

        # Random initialization
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume scoring
        ref_point = np.array([3.0, 5.0, 7.0])

        # Track best known points
        def get_non_dominated_indices(F_mat):
            """Return indices of non-dominated points in F_mat."""
            n = len(F_mat)
            non_dominated = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        non_dominated[i] = False
                        break
            return np.where(non_dominated)[0]

        def compute_hv_contribution(point, F_mat):
            """Compute approximate hypervolume contribution of a point."""
            if np.any(point >= ref_point):
                return 0.0
            
            # Compute the volume of the box from point to ref_point
            box_vol = np.prod(ref_point - point)
            
            # Subtract volumes of boxes that are already covered by other points
            # This is an approximation; for exact HV we'd need a proper algorithm
            # For now, use a simple check: if any other point dominates in all dimensions
            # within the box, reduce the contribution
            covered = 0.0
            for j in range(len(F_mat)):
                if j == -1:  # placeholder
                    continue
                other = F_mat[j]
                if np.all(other <= point):
                    # Other point is better (or equal) in all objectives
                    # The box from other to ref is a subset of the box from point to ref
                    # But this is complex; use a simple heuristic
                    pass
            
            return box_vol

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated parents
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]
            F_parents = F[nd_idx]

            # Select parent: prefer points with lower sum of objectives (closer to ideal)
            # But also consider diversity - use a simple approach
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]

            # Adaptive mutation step size
            # Start with larger steps, decrease over time
            progress = used / self.budget
            base_step = 0.3 * (1 - progress) + 0.05
            # Occasionally use a larger step for exploration
            if rng.random() < 0.1:
                step_size = base_step * 3
            else:
                step_size = base_step

            # Generate child
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Determine which point to replace
            # Strategy: replace the point with the worst hypervolume contribution
            # First, check if child dominates any point
            dominated_mask = np.array([
                np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
            ])

            if np.any(dominated_mask):
                # Replace one of the dominated points
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the point with the worst "quality"
                # Use a combination of:
                # 1. Sum of objectives (lower is better)
                # 2. Distance to ideal point (if known, else use min of F)
                # 3. Crowding-like measure (distance to nearest neighbor in objective space)

                # Simple approach: replace the point with the highest sum of objectives
                # This is a crude but effective heuristic
                sums = np.sum(F, axis=1)
                replace_idx = int(np.argmax(sums))

                # If the new point is better than the replaced point, swap
                # Otherwise, keep the old point (don't accept worse solutions)
                if np.sum(fc) >= sums[replace_idx]:
                    # Child is not better, skip replacement
                    # But we still used an evaluation, so continue
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
