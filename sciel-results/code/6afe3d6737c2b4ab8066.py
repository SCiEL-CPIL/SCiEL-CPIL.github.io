"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic resolution-aware grid search with adaptive refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_span = hi - lo
        n = self.dim
        n_obj = self.n_obj

        # Storage
        X = []
        F = []

        def evaluate(x):
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            return f

        # Phase 1: Coarse deterministic grid (Sobol-like via Halton)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = np.zeros((count, dim))
            for d in range(dim):
                base = d + 2
                for i in range(count):
                    points[i, d] = halton(i + 1, base)
            return points

        # Initial coarse grid: ~80 points
        n_init = 80
        init_pts = halton_sequence(n_init, n) * rng_span + lo
        for x in init_pts:
            if time.time() > deadline:
                break
            evaluate(x)

        # Also add corner points and midpoints for structure
        struct_pts = []
        for i in range(2 ** n):
            pt = np.array([lo[j] if (i >> j) & 1 else hi[j] for j in range(n)])
            struct_pts.append(pt)
        for j in range(n):
            mid = lo.copy()
            mid[j] = (lo[j] + hi[j]) / 2
            struct_pts.append(mid)
        for pt in struct_pts:
            if time.time() > deadline:
                break
            if len(X) >= self.budget:
                break
            evaluate(pt)

        # Phase 2: Adaptive refinement
        # Compute non-dominated set and refine around front
        while len(X) < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F)
            X_arr = np.array(X)

            # Find non-dominated points
            nondom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if not nondom_mask[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or not nondom_mask[j]:
                        continue
                    # j dominates i if j <= i in all and j < i in at least one
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom_mask[i] = False
                        break

            nondom_idx = np.where(nondom_mask)[0]
            if len(nondom_idx) == 0:
                nondom_idx = np.array([0])

            # Estimate curvature: look at gaps between consecutive non-dominated points
            # Sort non-dominated points by first objective
            nd_F = F_arr[nondom_idx]
            nd_X = X_arr[nondom_idx]
            order = np.argsort(nd_F[:, 0])
            nd_F = nd_F[order]
            nd_X = nd_X[order]

            # Compute gaps in objective space
            new_points = []
            # Add midpoints between consecutive non-dominated points in objective space
            for k in range(len(nd_F) - 1):
                mid_obj = (nd_F[k] + nd_F[k + 1]) / 2.0
                mid_x = (nd_X[k] + nd_X[k + 1]) / 2.0
                # Perturb slightly to avoid exact duplicate
                pert = rng.standard_normal(n) * 0.02 * rng_span
                new_x = np.clip(mid_x + pert, lo, hi)
                new_points.append(new_x)

            # Also add points slightly offset from each non-dominated point
            for k in range(len(nd_X)):
                for scale in [0.05, 0.1]:
                    pert = rng.standard_normal(n) * scale * rng_span
                    new_x = np.clip(nd_X[k] + pert, lo, hi)
                    new_points.append(new_x)

            # Also explore regions not yet covered: use Halton for next batch
            if len(new_points) < 10:
                extra = halton_sequence(min(10, self.budget - len(X)), n) * rng_span + lo
                new_points.extend([p for p in extra if len(new_points) + len(X) < self.budget])

            # Evaluate new points
            for new_x in new_points:
                if len(X) >= self.budget or time.time() > deadline:
                    break
                # Skip if too close to existing points
                too_close = False
                for existing in X_arr:
                    if np.linalg.norm(new_x - existing) < 0.01 * np.linalg.norm(rng_span):
                        too_close = True
                        break
                if not too_close:
                    evaluate(new_x)

            # If we're making slow progress, do a random restart in under-explored region
            if len(X) - n_init - len(struct_pts) < 5 and len(X) < self.budget - 20:
                random_pts = (lo + rng_span * rng.random((20, n)))
                for rp in random_pts:
                    if len(X) >= self.budget or time.time() > deadline:
                        break
                    evaluate(rp)

            # Final phase: local refinement around best hypervolume-contributing points
            if len(X) >= self.budget * 0.7:
                F_arr = np.array(F)
                # Find points that contribute most to hypervolume (simplification: non-dominated)
                nondom_mask = np.ones(len(F_arr), dtype=bool)
                for i in range(len(F_arr)):
                    if not nondom_mask[i]:
                        continue
                    for j in range(len(F_arr)):
                        if i == j or not nondom_mask[j]:
                            continue
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            nondom_mask[i] = False
                            break
                nondom_idx = np.where(nondom_mask)[0]
                if len(nondom_idx) > 0:
                    for k in nondom_idx[:5]:
                        for scale in [0.02, 0.05]:
                            if len(X) >= self.budget or time.time() > deadline:
                                break
                            pert = rng.standard_normal(n) * scale * rng_span
                            new_x = np.clip(X_arr[k] + pert, lo, hi)
                            evaluate(new_x)

        return X
# EVOLVE-BLOCK-END
