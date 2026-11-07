"""Multi-objective optimizer on RE34 with hypervolume-driven search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HV-guided multi-objective optimizer for RE34 (5 vars, 3 objectives)."""

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
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1705.0, 12.0, 0.3])

        def is_better_than_ref(f):
            return np.all(f < ref)

        # Efficient 2D hypervolume computation
        def hv_2d(points_2d, ref_2d):
            """Compute 2D hypervolume for points all < ref_2d."""
            m = len(points_2d)
            if m == 0:
                return 0.0
            if m == 1:
                return float((ref_2d[0] - points_2d[0][0]) * (ref_2d[1] - points_2d[0][1]))
            # Sort by first coordinate
            pts2 = points_2d[np.argsort(points_2d[:, 0])]
            hv = 0.0
            for i in range(m):
                p = pts2[i]
                x_next = pts2[i+1][0] if i+1 < m else ref_2d[0]
                y_min = np.min(pts2[i:, 1])
                hv += (x_next - p[0]) * (ref_2d[1] - y_min)
            return hv

        # 3D hypervolume using slice method
        def hv_3d(points, ref_point):
            """Compute 3D hypervolume for points all < ref_point."""
            n = len(points)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref_point - points[0]))
            # Sort by first coordinate
            pts = points[np.argsort(points[:, 0])]
            hv = 0.0
            for i in range(n):
                x_i = pts[i][0]
                x_next = pts[i+1][0] if i+1 < n else ref_point[0]
                # Active points in this x-slice: indices 0..i
                yz = pts[:i+1, 1:3]
                hv += (x_next - x_i) * hv_2d(yz, ref_point[1:3])
            return hv

        # Pareto dominance check
        def is_pareto(F_arr):
            """Return mask of Pareto-optimal points."""
            n = len(F_arr)
            if n == 0:
                return np.array([], dtype=bool)
            pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not pareto[i]:
                    continue
                for j in range(n):
                    if i == j or not pareto[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        pareto[i] = False
                        break
            return pareto

        # HV contribution of a single point
        def hv_contribution(point, pareto_set, ref_point):
            if len(pareto_set) == 0:
                return float(np.prod(ref_point - point))
            with_pt = np.vstack([pareto_set, [point]])
            mask_with = np.all(with_pt < ref_point, axis=1)
            with_pt = with_pt[mask_with]
            mask_without = np.all(pareto_set < ref_point, axis=1)
            without_pt = pareto_set[mask_without]
            hv_with = hv_3d(with_pt, ref_point) if len(with_pt) > 0 else 0.0
            hv_without = hv_3d(without_pt, ref_point) if len(without_pt) > 0 else 0.0
            return hv_with - hv_without

        # Seeded RNG for reproducibility
        rng = np.random.RandomState(42)

        # Initial population: Latin Hypercube Sampling
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(pop_size)) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Filter valid points
        valid_mask = np.array([is_better_than_ref(f) for f in F])
        X_valid = X[valid_mask]
        F_valid = F[valid_mask]

        # Maintain Pareto set
        pareto_X = X_valid.copy()
        pareto_F = F_valid.copy()

        # Main loop: HV-guided differential evolution
        while used < budget:
            if time.time() > deadline:
                break

            # Get current Pareto set
            if len(pareto_F) > 0:
                p_mask = is_pareto(pareto_F)
                pareto_F_cur = pareto_F[p_mask]
                pareto_X_cur = pareto_X[p_mask]
            else:
                pareto_F_cur = pareto_F
                pareto_X_cur = pareto_X

            # Select parents based on HV contribution
            if len(pareto_F_cur) >= 2:
                # Compute HV contributions
                contribs = np.zeros(len(pareto_F_cur))
                for i in range(len(pareto_F_cur)):
                    contribs[i] = hv_contribution(pareto_F_cur[i], pareto_F_cur, ref)
                contribs = np.clip(contribs, 1e-10, None)
                probs = contribs / contribs.sum()
                parent_idx = rng.choice(len(pareto_F_cur), p=probs)
                parent = pareto_X_cur[parent_idx]

                # Differential evolution: pick two other parents
                other_indices = np.delete(np.arange(len(pareto_F_cur)), parent_idx)
                other_indices = rng.choice(other_indices, size=min(2, len(other_indices)), replace=False)
                if len(other_indices) >= 2:
                    p2 = pareto_X_cur[other_indices[0]]
                    p3 = pareto_X_cur[other_indices[1]]
                    F_factor = 0.8 * (1 - used / budget) + 0.3
                    child = p2 + F_factor * (p2 - p3) + rng.standard_normal(dim) * 0.1 * (hi - lo)
                else:
                    child = parent + rng.standard_normal(dim) * 0.3 * (hi - lo)
            elif len(pareto_F_cur) == 1:
                parent = pareto_X_cur[0]
                child = parent + rng.standard_normal(dim) * 0.3 * (hi - lo)
            else:
                # No valid points, random exploration
                child = lo + rng.random(dim) * (hi - lo)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update archives
            if is_better_than_ref(fc):
                pareto_X = np.vstack([pareto_X, [child]])
                pareto_F = np.vstack([pareto_F, [fc]])

            # Prune Pareto set periodically
            if len(pareto_F) > 80:
                p_mask = is_pareto(pareto_F)
                pareto_X = pareto_X[p_mask]
                pareto_F = pareto_F[p_mask]
                if len(pareto_F) > 60:
                    # Keep top by HV contribution
                    contribs = np.zeros(len(pareto_F))
                    for i in range(len(pareto_F)):
                        contribs[i] = hv_contribution(pareto_F[i], pareto_F, ref)
                    top_idx = np.argsort(contribs)[-60:]
                    pareto_X = pareto_X[top_idx]
                    pareto_F = pareto_F[top_idx]

        return X

# EVOLVE-BLOCK-END
