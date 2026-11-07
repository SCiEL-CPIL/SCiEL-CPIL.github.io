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
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1705.0, 12.0, 0.3])
        rng = np.random.RandomState(42)

        # --- Hypervolume helpers ---
        def hv_2d(pts, ref2):
            """2D hypervolume. pts: (m,2), ref2: (2,). All pts < ref2."""
            m = len(pts)
            if m == 0:
                return 0.0
            if m == 1:
                return float((ref2[0] - pts[0, 0]) * (ref2[1] - pts[0, 1]))
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            hv = 0.0
            for i in range(m):
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < m else ref2[0]
                y_min = np.min(pts[i:, 1])
                hv += (x_next - x_cur) * (ref2[1] - y_min)
            return hv

        def hv_3d(pts, ref3):
            """3D hypervolume via slicing. pts: (n,3), ref3: (3,). All pts < ref3."""
            n = len(pts)
            if n == 0:
                return 0.0
            if n == 1:
                return float(np.prod(ref3 - pts[0]))
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            hv = 0.0
            for i in range(n):
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref3[0]
                yz = pts[:i + 1, 1:3]
                hv += (x_next - x_cur) * hv_2d(yz, ref3[1:3])
            return hv

        def is_valid(f):
            return np.all(f < ref)

        def is_pareto(F_arr):
            """Return boolean mask of Pareto-optimal points in F_arr."""
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

        def hv_contribution_single(point, pareto_F, ref3):
            """HV with point minus HV without point."""
            if len(pareto_F) == 0:
                return float(np.prod(ref3 - point))
            with_pt = np.vstack([pareto_F, [point]])
            mask_w = np.all(with_pt < ref3, axis=1)
            with_pt = with_pt[mask_w]
            mask_wo = np.all(pareto_F < ref3, axis=1)
            without_pt = pareto_F[mask_wo]
            hv_w = hv_3d(with_pt, ref3) if len(with_pt) > 0 else 0.0
            hv_wo = hv_3d(without_pt, ref3) if len(without_pt) > 0 else 0.0
            return hv_w - hv_wo

        # --- Initial sampling: Latin Hypercube ---
        pop_size = min(30, budget)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.arange(pop_size)
            rng.shuffle(perm)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(pop_size)) / pop_size
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Filter valid
        valid_mask = np.array([is_valid(f) for f in F])
        X_valid = X[valid_mask]
        F_valid = F[valid_mask]

        # Main archive (all valid points found so far)
        archive_X = X_valid.copy()
        archive_F = F_valid.copy()

        # --- Main loop ---
        while used < budget:
            if time.time() > deadline:
                break

            # Get current Pareto front
            if len(archive_F) > 0:
                p_mask = is_pareto(archive_F)
                pareto_X = archive_X[p_mask]
                pareto_F = archive_F[p_mask]
            else:
                pareto_X = np.empty((0, dim))
                pareto_F = np.empty((0, n_obj))

            # Select parent
            if len(pareto_F) > 0:
                # Compute HV contributions
                contribs = np.zeros(len(pareto_F))
                for i in range(len(pareto_F)):
                    contribs[i] = hv_contribution_single(pareto_F[i], pareto_F, ref)
                contribs = np.clip(contribs, 1e-12, None)
                probs = contribs / contribs.sum()
                parent_idx = rng.choice(len(pareto_F), p=probs)
                parent = pareto_X[parent_idx]
            else:
                parent = X[rng.randint(len(X))]

            # Adaptive mutation
            progress = used / budget
            # Larger steps early, smaller later
            step_scale = 0.4 * (1 - progress) + 0.03
            child = parent + step_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            if is_valid(fc):
                archive_X = np.vstack([archive_X, [child]])
                archive_F = np.vstack([archive_F, [fc]])

            # Prune archive to keep it manageable (keep Pareto + top HV)
            if len(archive_F) > 150:
                p_mask = is_pareto(archive_F)
                if p_mask.sum() < len(archive_F):
                    archive_X = archive_X[p_mask]
                    archive_F = archive_F[p_mask]
                if len(archive_F) > 100:
                    # Keep top 100 by HV contribution
                    contribs = np.zeros(len(archive_F))
                    for i in range(len(archive_F)):
                        contribs[i] = hv_contribution_single(archive_F[i], archive_F, ref)
                    top_idx = np.argsort(contribs)[-100:]
                    archive_X = archive_X[top_idx]
                    archive_F = archive_F[top_idx]

        # Return all evaluated points (return value is ignored by evaluator)
        return archive_X

# EVOLVE-BLOCK-END
