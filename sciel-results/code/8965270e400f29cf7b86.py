"""Multi-objective optimizer for RE34 using hypervolume-driven simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective Simulated Annealing with a hypervolume-driven acceptance
    criterion and a non-dominated archive.

    The search treats the process thermodynamically: a current solution is
    mutated, and the move is accepted if it is non-dominated, or with a
    temperature-dependent probability when it is dominated.  Acceptance is
    biased by the hypervolume contribution of the candidate to the archive so
    that moves which expand the non-dominated set's volume are preferred.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # safety margin before the 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        rng = np.random.default_rng(12345)

        # ---- helpers -------------------------------------------------------
        def hv_contribution(pts, r):
            """Exact hypervolume of pts (n x 3) dominated by reference r.
            Only points strictly better than r in every objective count."""
            pts = np.asarray(pts, dtype=float)
            if pts.size == 0:
                return 0.0
            mask = np.all(pts < r, axis=1)
            pts = pts[mask]
            if pts.shape[0] == 0:
                return 0.0
            # keep only non-dominated among pts (w.r.t. minimization)
            nd = []
            for i in range(pts.shape[0]):
                dominated = False
                for j in range(pts.shape[0]):
                    if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            pts = pts[nd]
            n = pts.shape[0]
            if n == 0:
                return 0.0
            if n == 1:
                p = pts[0]
                if np.any(p >= r):
                    return 0.0
                return float(np.prod(r - p))
            if n == 2:
                a, b = pts[0], pts[1]
                # sort by first objective
                if a[0] > b[0]:
                    a, b = b, a
                # a dominates in obj0; contribution = (r0-a0)*(r1-min(a1,b1))*(r2-min(a2,b2))
                # plus region only b covers: (b0-a0)*(r1-b1)*(r2-b2) if b1,b2 < r
                vol = 0.0
                m1 = min(a[1], b[1]); m2 = min(a[2], b[2])
                if a[0] < r[0] and m1 < r[1] and m2 < r[2]:
                    vol += (r[0] - a[0]) * (r[1] - m1) * (r[2] - m2)
                if b[0] < r[0] and b[1] < r[1] and b[2] < r[2]:
                    vol += (b[0] - a[0]) * (r[1] - b[1]) * (r[2] - b[2])
                return float(max(0.0, vol))
            # general n: O(n^3) inclusion via sorting + 2D HV, fine for small n
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # 2D hypervolume of the projection (obj1,obj2) weighted by obj0 gaps
            total = 0.0
            for k in range(n):
                p = pts[k]
                if p[0] >= r[0]:
                    break
                # x-extent: from p[0] to next point's obj0 (or r[0])
                next0 = pts[k + 1][0] if k + 1 < n else r[0]
                dx = next0 - p[0]
                if dx <= 0:
                    continue
                # 2D HV of points k..n-1 in (obj1,obj2), reference (r1,r2)
                sub = pts[k:, 1:]
                # filter strictly better
                sub = sub[np.all(sub < r[1:], axis=1)]
                if sub.shape[0] == 0:
                    continue
                # 2D exact HV
                sub = sub[np.all(sub < r[1:], axis=1)]
                # remove dominated in 2D
                nd2 = []
                for i in range(sub.shape[0]):
                    dom = False
                    for j in range(sub.shape[0]):
                        if i != j and np.all(sub[j] <= sub[i]) and np.any(sub[j] < sub[i]):
                            dom = True
                            break
                    if not dom:
                        nd2.append(i)
                sub = sub[nd2]
                m = sub.shape[0]
                if m == 0:
                    continue
                # 2D HV by sorting on first coord
                sub = sub[np.argsort(sub[:, 0])]
                hv2 = 0.0
                prev = sub[0][0]
                maxy = -np.inf
                for i in range(m):
                    x1, y1 = sub[i]
                    if y1 > maxy:
                        maxy = y1
                        # width from prev to x1
                        if x1 > prev:
                            if maxy < r[2]:
                                hv2 += (x1 - prev) * (r[2] - maxy)
                        prev = x1
                # final segment
                if prev < r[1] and maxy < r[2]:
                    hv2 += (r[1] - prev) * (r[2] - maxy)
                total += dx * hv2
            return float(max(0.0, total))

        def is_dominated(f, archive):
            """True if archive (n x 3) dominates f."""
            if archive.shape[0] == 0:
                return False
            return bool(np.any(np.all(archive <= f, axis=1) & np.any(archive < f, axis=1)))

        def archive_dominated_by(f, archive):
            """Indices of archive points dominated by f."""
            if archive.shape[0] == 0:
                return []
            return np.where(np.all(f <= archive, axis=1) & np.any(f < archive, axis=1))[0].tolist()

        # ---- archive -------------------------------------------------------
        archive = []  # list of (x, f)

        def add_to_archive(x, f):
            """Add (x,f) to archive, removing points it dominates. Return True if added."""
            if is_dominated(f, np.array(archive_f) if archive else np.empty((0, 3))):
                return False
            # remove dominated archive points
            dom_idx = archive_dominated_by(f, np.array(archive_f) if archive else np.empty((0, 3)))
            for i in sorted(dom_idx, reverse=True):
                archive.pop(i)
            archive.append((x.copy(), f.copy()))
            return True

        archive_f = np.empty((0, 3))

        def refresh_archive_f():
            nonlocal archive_f
            archive_f = np.array([f for _, f in archive]) if archive else np.empty((0, 3))

        # ---- initial population -------------------------------------------
        n_init = 40
        X_init = lo + span * rng.random((n_init, dim))
        used = 0
        for x in X_init:
            if time.time() > deadline:
                break
            f = np.asarray(problem(x), dtype=float)
            used += 1
            add_to_archive(x, f)
        refresh_archive_f()

        # current solution: pick an archive point (or a random one)
        if archive:
            cur_x, cur_f = archive[0]
        else:
            cur_x = lo + span * rng.random(dim)
            cur_f = np.asarray(problem(cur_x), dtype=float)
            used += 1

        # ---- annealing loop -----------------------------------------------
        T0 = 1.0
        T_end = 1e-6
        n_iter = max(1, budget - used)
        # geometric cooling schedule
        if n_iter > 1:
            cool = (T_end / T0) ** (1.0 / n_iter)
        else:
            cool = T_end / T0
        T = T0

        for _ in range(n_iter):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # mutation: Gaussian step, scale decays mildly
            sigma = 0.35 * (0.5 + 0.5 * (T / T0))
            cand_x = cur_x + sigma * span * rng.standard_normal(dim)
            cand_x = np.clip(cand_x, lo, hi)
            cand_f = np.asarray(problem(cand_x), dtype=float)
            used += 1

            # acceptance decision
            if not is_dominated(cand_f, archive_f):
                # non-dominated: accept if it adds hypervolume or is new
                add_to_archive(cand_x, cand_f)
                refresh_archive_f()
                cur_x, cur_f = cand_x, cand_f
            else:
                # dominated: hypervolume-driven + temperature acceptance
                # compute hypervolume with and without candidate
                base_hv = hv_contribution(archive_f, ref)
                trial = np.vstack([archive_f, cand_f[None, :]]) if archive_f.shape[0] else cand_f[None, :]
                trial_hv = hv_contribution(trial, ref)
                d_hv = trial_hv - base_hv
                # accept probability: boosted by positive hypervolume gain and temperature
                if d_hv > 0:
                    p_acc = min(1.0, 0.2 + 0.8 * (d_hv / (base_hv + 1e-12)) + 0.5 * (T / T0))
                else:
                    p_acc = 0.05 * (T / T0)
                if rng.random() < p_acc:
                    # accept but only keep in archive if non-dominated
                    add_to_archive(cand_x, cand_f)
                    refresh_archive_f()
                    cur_x, cur_f = cand_x, cand_f

            T *= cool

        return cur_x
# EVOLVE-BLOCK-END
