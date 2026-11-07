"""Multi-objective optimizer on RE21 using DE with hypervolume-based selection."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive mutation and hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, self.budget // 5)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume calculation
        ref_point = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref):
            """Calculate hypervolume for 2D points dominated by reference point."""
            # Filter points that are better than reference in all objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                if y1 < prev_y:
                    hv += (ref[0] - x1) * (prev_y - y1)
                    prev_y = y1
            return hv

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Adaptive DE parameters
        F_mut = 0.8
        CR = 0.9

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated points
            nd_mask = non_dominated_mask(F)
            n_nd = np.sum(nd_mask)

            # If we have few non-dominated points, use all as potential parents
            if n_nd < 3:
                parent_indices = np.arange(len(F))
            else:
                parent_indices = np.where(nd_mask)[0]

            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct parents
                n_p = len(parent_indices)
                if n_p < 3:
                    # If not enough non-dominated, use all
                    parent_indices = np.arange(len(F))
                    n_p = len(parent_indices)

                idx1, idx2, idx3 = rng.choice(n_p, size=3, replace=False)
                p1, p2, p3 = parent_indices[idx1], parent_indices[idx2], parent_indices[idx3]

                # Mutation
                mutant = F[p1] + F_mut * (F[p2] - F[p3])
                # Actually we need to work in decision space, not objective space
                # Let's redo this properly
                x1, x2, x3 = X[p1], X[p2], X[p3]
                mutant_x = x1 + F_mut * (x2 - x3)

                # Crossover
                trial = np.empty(self.dim)
                jrand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = mutant_x[j]
                    else:
                        trial[j] = X[p1][j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Selection: keep better of trial and original
                # Use hypervolume contribution as a simple criterion
                # If trial dominates original, replace
                if np.all(f_trial <= F[p1]) and np.any(f_trial < F[p1]):
                    X[p1] = trial
                    F[p1] = f_trial
                elif not (np.all(F[p1] <= f_trial) and np.any(F[p1] < f_trial)):
                    # Neither dominates; use hypervolume contribution
                    # Simple heuristic: if trial has better sum of normalized objectives
                    norm_trial = np.sum(f_trial / (ref_point + 1e-10))
                    norm_orig = np.sum(F[p1] / (ref_point + 1e-10))
                    if norm_trial < norm_orig:
                        X[p1] = trial
                        F[p1] = f_trial

            # Adaptive mutation rate
            F_mut = 0.5 + 0.5 * np.exp(-used / 50.0)

        return X
