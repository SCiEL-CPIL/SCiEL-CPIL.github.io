import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        eval_count = 0
        max_budget = self.budget

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= max_budget or not time_ok():
                return None
            eval_count += 1
            try:
                return np.asarray(problem(x), dtype=float).reshape(-1)[:n_obj]
            except Exception:
                return None

        # ---------- Pareto / hypervolume helpers ----------
        def non_dominated_indices(F):
            F = np.asarray(F)
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
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
            return np.where(mask)[0]

        def hypervolume_3d(F, ref_pt):
            """Exact 3D hypervolume of non-dominated set F w.r.t. ref_pt.
            Only points strictly better than ref_pt in every objective count."""
            F = np.asarray(F, dtype=float)
            if len(F) == 0:
                return 0.0
            keep = np.all(F < ref_pt, axis=1)
            F = F[keep]
            if len(F) == 0:
                return 0.0
            idx = non_dominated_indices(F)
            F = F[idx]
            n = len(F)
            if n == 0:
                return 0.0

            # Sort by objective 0 ascending
            order = np.argsort(F[:, 0])
            P = F[order]
            total = 0.0
            for i in range(n):
                p = P[i]
                if i + 1 < n:
                    next0 = P[i + 1][0]
                else:
                    next0 = ref_pt[0]
                w0 = next0 - p[0]
                if w0 <= 0:
                    continue
                p1, p2 = p[1], p[2]
                r1, r2 = ref_pt[1], ref_pt[2]
                # 2D area of rectangle [p1, r1] x [p2, r2] minus overlaps
                area = (r1 - p1) * (r2 - p2)
                # Subtract overlaps with earlier points that block this area
                for j in range(i):
                    q = P[j]
                    if q[1] < p1 and q[2] < p2:
                        # Overlap region in obj1: [q[1], p1], in obj2: [q[2], p2]
                        # But we need the union of all such overlaps
                        # For correctness, compute union of rectangles
                        pass
                
                # More careful: compute the 2D hypervolume of points in obj1-obj2
                # that are "below" p (i.e., q[1] < p1 and q[2] < p2)
                # The area to subtract is the 2D hypervolume of these points
                # w.r.t. reference (p1, p2) in the obj1-obj2 plane
                below = [(P[j][1], P[j][2]) for j in range(i) if P[j][1] < p1 and P[j][2] < p2]
                if below:
                    below_arr = np.array(below)
                    # 2D hypervolume of below_arr w.r.t. (p1, p2)
                    below_arr = below_arr[below_arr[:, 0] < p1]
                    below_arr = below_arr[below_arr[:, 1] < p2]
                    if len(below_arr) > 0:
                        # Non-dominated in 2D
                        n2 = len(below_arr)
                        nd2 = np.ones(n2, dtype=bool)
                        for a in range(n2):
                            if not nd2[a]:
                                continue
                            for b in range(n2):
                                if a == b or not nd2[b]:
                                    continue
                                if below_arr[b][0] <= below_arr[a][0] and below_arr[b][1] <= below_arr[a][1] and (below_arr[b][0] < below_arr[a][0] or below_arr[b][1] < below_arr[a][1]):
                                    nd2[a] = False
                                    break
                        below_arr = below_arr[nd2]
                        # Sort by obj1 ascending
                        below_arr = below_arr[np.argsort(below_arr[:, 0])]
                        hv2 = 0.0
                        for k in range(len(below_arr)):
                            bk = below_arr[k]
                            if k + 1 < len(below_arr):
                                next1 = below_arr[k + 1][0]
                            else:
                                next1 = p1
                            w1 = next1 - bk[0]
                            if w1 <= 0:
                                continue
                            # Height in obj2: from bk[1] to p2, minus overlaps
                            h2 = p2 - bk[1]
                            # Subtract overlaps with earlier points in this 2D set
                            for m in range(k):
                                bm = below_arr[m]
                                if bm[1] < bk[1]:
                                    h2 = min(h2, p2 - bm[1])
                            if h2 > 0:
                                hv2 += w1 * h2
                        area -= hv2
                
                if area > 0:
                    total += w0 * area
            return total

        # ---------- Main optimization loop ----------
        X = []
        F = []

        # Phase 1: LHS initialization for broad coverage
        n_initial = min(30, max_budget)
        if n_initial > 0:
            samples = np.zeros((n_initial, dim))
            for d in range(dim):
                perm = np.random.permutation(n_initial)
                samples[:, d] = (perm + np.random.random(n_initial)) / n_initial
            samples = lo + (hi - lo) * samples
            for x in samples:
                if not time_ok():
                    break
                f = evaluate(x)
                if f is None:
                    break
                X.append(x)
                F.append(f)

        if len(X) == 0:
            x0 = lo + 0.5 * (hi - lo)
            f0 = evaluate(x0)
            if f0 is not None:
                X.append(x0)
                F.append(f0)
            return X[0] if len(X) > 0 else x0

        X = np.array(X)
        F = np.array(F)

        # Track decision-to-objective mapping
        all_X = list(X)
        all_F = list(F)

        # Phase 2: Hypervolume-driven adaptive sampling with local refinement
        # Maintain the non-dominated set
        nd_idx = non_dominated_indices(F)
        nd_F = F[nd_idx].copy() if len(nd_idx) > 0 else F.copy()

        # Local search around promising points
        def local_refine(x_start, f_start, nd_F_current, n_steps=8):
            """Refine a point using hypervolume-guided local search."""
            x_best = x_start.copy()
            f_best = f_start.copy()
            base_hv = hypervolume_3d(nd_F_current, ref)
            base_nd = nd_F_current.copy()

            for step in range(n_steps):
                if not time_ok() or eval_count >= max_budget:
                    break
                # Adaptive step size
                scale = 0.15 * (1.0 - step / float(max(1, n_steps))) ** 0.5
                x_new = x_best + scale * np.random.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                all_X.append(x_new)
                all_F.append(f_new)
                # Check if f_new improves hypervolume
                cand = np.vstack([base_nd, f_new])
                hv = hypervolume_3d(cand, ref)
                if hv > base_hv + 1e-12:
                    x_best = x_new
                    f_best = f_new
                    nd_idx2 = non_dominated_indices(cand)
                    base_nd = cand[nd_idx2]
                    base_hv = hv
            return x_best, f_best, base_nd, base_hv

        # Phase 2a: Refine each non-dominated point
        remaining = max_budget - eval_count
        if len(nd_F) > 0 and remaining > 0:
            steps_per_point = max(4, remaining // (len(nd_F) + 2))
            for i in range(len(nd_F)):
                if not time_ok() or eval_count >= max_budget:
                    break
                # Find the decision-space point corresponding to nd_F[i]
                x_start = None
                for k in range(len(all_X)):
                    if np.allclose(all_F[k], nd_F[i], atol=1e-10):
                        x_start = all_X[k]
                        break
                if x_start is None:
                    x_start = lo + np.random.random(dim) * (hi - lo)
                f_start = nd_F[i]
                x_ref, f_ref, nd_F, base_hv = local_refine(x_start, f_start, nd_F, steps_per_point)

        # Phase 2b: Gap-filling with local refinement
        # Compute gaps in objective space and sample near gap centers
        remaining = max_budget - eval_count
        if len(nd_F) >= 2 and remaining > 0:
            n_gaps = min(remaining // 6, 20)
            for _ in range(n_gaps):
                if not time_ok() or eval_count >= max_budget:
                    break
                # Pick two random non-dominated points and interpolate in objective space
                i, j = np.random.choice(len(nd_F), 2, replace=False)
                t = np.random.random()
                # Interpolate in decision space
                x_i = None
                x_j = None
                for k in range(len(all_X)):
                    if np.allclose(all_F[k], nd_F[i], atol=1e-10):
                        x_i = all_X[k]
                        break
                for k in range(len(all_X)):
                    if np.allclose(all_F[k], nd_F[j], atol=1e-10):
                        x_j = all_X[k]
                        break
                if x_i is not None and x_j is not None:
                    x_new = t * x_i + (1 - t) * x_j + 0.05 * np.random.standard_normal(dim)
                else:
                    x_new = lo + np.random.random(dim) * (hi - lo)
                x_new = np.clip(x_new, lo, hi)
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                all_X.append(x_new)
                all_F.append(f_new)
                # Update non-dominated set
                cand = np.vstack([nd_F, f_new])
                hv = hypervolume_3d(cand, ref)
                nd_idx2 = non_dominated_indices(cand)
                nd_F = cand[nd_idx2]
                # If it's non-dominated, refine it
                if any(np.allclose(f_new, nd_F[k], atol=1e-10) for k in range(len(nd_F))):
                    x_ref, f_ref, nd_F, _ = local_refine(x_new, f_new, nd_F, 4)

        # Phase 2c: Random restarts with local refinement
        remaining = max_budget - eval_count
        n_restarts = min(remaining // 10, 15)
        for _ in range(n_restarts):
            if not time_ok() or eval_count >= max_budget:
                break
            x_rand = lo + np.random.random(dim) * (hi - lo)
            f_rand = evaluate(x_rand)
            if f_rand is None:
                break
            all_X.append(x_rand)
            all_F.append(f_rand)
            # Check if it improves hypervolume
            cand = np.vstack([nd_F, f_rand])
            hv = hypervolume_3d(cand, ref)
            nd_idx2 = non_dominated_indices(cand)
            nd_F = cand[nd_idx2]
            # If it's non-dominated, refine it
            if any(np.allclose(f_rand, nd_F[k], atol=1e-10) for k in range(len(nd_F))):
                x_ref, f_ref, nd_F, _ = local_refine(x_rand, f_rand, nd_F, 6)

        # Final: return the best point (first non-dominated point)
        if len(all_X) > 0:
            final_nd = non_dominated_indices(np.array(all_F))
            if len(final_nd) > 0:
                return all_X[final_nd[0]]
            return all_X[0]
        return lo + 0.5 * (hi - lo)
