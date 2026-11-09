"""Multi-objective optimizer for WFG4 using grid-based decomposition with LHS."""
import numpy as np
import time
import math
import sys

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        mid = 0.5 * (lo + hi)
        budget = self.budget

        eval_count = 0
        # Archive: list of (x, f) for all evaluated points
        archive_X = []
        archive_F = []

        def time_left():
            return deadline - time.time()

        def check_deadline():
            if time.time() > deadline:
                raise TimeoutError()

        def evaluate(x):
            nonlocal eval_count
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            archive_X.append(x.copy())
            archive_F.append(f.copy())
            return f

        # ---- Hypervolume computation (3D) ----
        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3D points dominated by ref."""
            if len(points) == 0:
                return 0.0
            pts = [tuple(p) for p in points if all(p[i] < ref[i] for i in range(3))]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective (ascending)
            pts.sort(key=lambda p: p[0])
            # Remove dominated points
            non_dom = []
            best_1 = float('inf')
            best_2 = float('inf')
            for p in pts:
                if p[1] < best_1 and p[2] < best_2:
                    non_dom.append(p)
                    if p[1] < best_1:
                        best_1 = p[1]
                    if p[2] < best_2:
                        best_2 = p[2]
            # For 3D, use the standard sweep
            # Sort by first coordinate
            non_dom.sort(key=lambda p: p[0])
            # Remove points dominated by others in 2D (obj1, obj2)
            # Actually, let's just do the standard 3D HV
            hv = 0.0
            n = len(non_dom)
            for i in range(n):
                p = non_dom[i]
                # Width in obj0: from p[0] to next point's obj0 (or ref[0])
                if i + 1 < n:
                    w0 = non_dom[i+1][0] - p[0]
                else:
                    w0 = ref[0] - p[0]
                if w0 <= 0:
                    continue
                # In the slice [p[0], p[0]+w0], compute 2D HV of points with obj0 <= p[0]+w0
                # For points in this slice, their obj0 is all <= p[0]+w0
                # The 2D HV is over (obj1, obj2)
                # Points that are active in this slice: those with obj0 <= p[0] + w0
                # Since we sorted, all points up to index i have obj0 <= p[0]
                # Points from i+1 onward have obj0 >= non_dom[i+1][0] = p[0] + w0
                # So only points 0..i are active
                # Compute 2D HV of points 0..i projected to (obj1, obj2)
                pts_2d = [(non_dom[j][1], non_dom[j][2]) for j in range(i+1)]
                hv_2d = hypervolume_2d(pts_2d, (ref[1], ref[2]))
                hv += w0 * hv_2d
            return hv

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume."""
            if len(points) == 0:
                return 0.0
            pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if len(pts) == 0:
                return 0.0
            # Sort by first coordinate
            pts.sort(key=lambda p: p[0])
            # Remove dominated
            non_dom = []
            best_y = float('inf')
            for p in pts:
                if p[1] < best_y:
                    non_dom.append(p)
                    best_y = p[1]
            hv = 0.0
            for i in range(len(non_dom)):
                p = non_dom[i]
                if i + 1 < len(non_dom):
                    w = non_dom[i+1][0] - p[0]
                else:
                    w = ref[0] - p[0]
                if w <= 0:
                    continue
                h = ref[1] - p[1]
                hv += w * h
            return hv

        def compute_hv():
            """Compute HV of current archive with reference point."""
            ref = np.array([3.0, 5.0, 7.0])
            F = np.array(archive_F)
            return hypervolume_3d(F.tolist(), ref.tolist())

        # ---- Phase 1: LHS sampling for broad spatial coverage ----
        n_lhs = min(80, budget)
        lhs_indices = []
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_indices.append(perm)

        for i in range(n_lhs):
            if eval_count >= budget or time_left() < 5.0:
                break
            x = np.zeros(dim)
            for d in range(dim):
                stratum_idx = lhs_indices[d][i]
                frac = (stratum_idx + np.random.uniform()) / n_lhs
                x[d] = lo[d] + frac * span[d]
            evaluate(x)

        # ---- Phase 2: Analyze LHS results to find front structure ----
        F_arr = np.array(archive_F)
        ref = np.array([3.0, 5.0, 7.0])
        valid_mask = np.all(F_arr < ref, axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            valid_F = F_arr[valid_indices]
            valid_X = [archive_X[i] for i in valid_indices]
            # Find non-dominated subset
            nd_idx = []
            for i in range(len(valid_F)):
                is_dom = False
                for j in range(len(valid_F)):
                    if i != j and np.all(valid_F[j] <= valid_F[i]) and np.any(valid_F[j] < valid_F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_idx.append(i)
            if nd_idx:
                valid_F = valid_F[nd_idx]
                valid_X = [valid_X[i] for i in nd_idx]
        else:
            valid_F = F_arr
            valid_X = archive_X

        # Determine front variable (largest spread among valid points)
        if len(valid_X) > 1:
            front_var = 0
            best_spread = 0
            for i in range(dim):
                vals = np.array([vx[i] for vx in valid_X])
                spread = np.max(vals) - np.min(vals)
                if spread > best_spread:
                    best_spread = spread
                    front_var = i
        else:
            front_var = 0

        # ---- Phase 3: Grid-based refinement along the front ----
        if len(valid_X) > 1:
            optimal_penalty = np.array([np.median([vx[i] for vx in valid_X]) for i in range(dim)])
        else:
            optimal_penalty = mid.copy()

        n_grid = min(25, budget - eval_count)
        if n_grid > 0 and time_left() > 5.0:
            grid_ts = np.linspace(0.0, 1.0, n_grid)
            for t in grid_ts:
                if eval_count >= budget or time_left() < 2.0:
                    break
                x = optimal_penalty.copy()
                x[front_var] = lo[front_var] + t * span[front_var]
                evaluate(x)

        # ---- Phase 4: Local optimization around grid points ----
        remaining = budget - eval_count
        if remaining > 10 and time_left() > 10.0:
            grid_start = len(archive_X) - n_grid
            if grid_start < 0:
                grid_start = 0
            n_refine = min(n_grid, max(1, remaining // (dim * 2)))
            evals_per_point = max(3, remaining // max(1, n_refine))

            for pi in range(n_refine):
                if eval_count >= budget or time_left() < 5.0:
                    break
                idx = grid_start + pi
                if idx >= len(archive_X):
                    break
                x_cur = archive_X[idx].copy()
                f_cur = archive_F[idx].copy()

                for iteration in range(evals_per_point // dim):
                    if eval_count >= budget or time_left() < 2.0:
                        break
                    improved = False
                    for i in range(dim):
                        if i == front_var:
                            continue
                        if eval_count >= budget or time_left() < 1.0:
                            break
                        for delta_frac in [0.1, -0.1, 0.2, -0.2]:
                            if eval_count >= budget or time_left() < 0.5:
                                break
                            x_trial = x_cur.copy()
                            x_trial[i] += delta_frac * span[i]
                            x_trial[i] = np.clip(x_trial[i], lo[i], hi[i])
                            if np.allclose(x_trial, x_cur, atol=1e-12):
                                continue
                            f_trial = evaluate(x_trial)
                            if np.sum(f_trial) < np.sum(f_cur):
                                if np.max(f_trial - f_cur) < 0.05:
                                    x_cur = x_trial.copy()
                                    f_cur = f_trial.copy()
                                    improved = True
                                    break
                if not any(np.allclose(x_cur, ax, atol=1e-10) for ax in archive_X):
                    archive_X.append(x_cur.copy())
                    archive_F.append(f_cur.copy())

        # ---- Phase 5: Final dense sweep with best penalty configuration ----
        remaining = budget - eval_count
        if remaining > 5 and time_left() > 2.0:
            F_arr = np.array(archive_F)
            valid_mask = np.all(F_arr < ref, axis=1)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 1:
                valid_X = [archive_X[i] for i in valid_indices]
                final_penalty = np.array([np.median([vx[i] for vx in valid_X]) for i in range(dim)])
            else:
                final_penalty = optimal_penalty.copy()

            n_final = min(remaining, 50)
            final_ts = np.linspace(0.0, 1.0, n_final)
            for t in final_ts:
                if eval_count >= budget or time_left() < 1.0:
                    break
                x = final_penalty.copy()
                x[front_var] = lo[front_var] + t * span[front_var]
                evaluate(x)

        return np.array(archive_X) if archive_X else np.array([mid])
