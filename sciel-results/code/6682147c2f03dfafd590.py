# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        t0 = time.time()
        deadline = t0 + 270.0   # safety margin before the 330s kill

        def valid(f):
            return np.all(np.isfinite(f)) & np.all(f < ref)

        # ---------------- diversity-seeded population ----------------
        rng = np.random.default_rng(0)
        init = min(budget, 80)
        rows = []
        # corners
        for mask in range(1 << dim):
            x = lo.copy()
            for i in range(dim):
                if mask & (1 << i):
                    x[i] = hi[i]
            rows.append(x)
        # Latin hypercube
        n_lhs = max(0, init - len(rows))
        if n_lhs:
            for j in range(dim):
                perm = rng.permutation(n_lhs)
                u = (perm + rng.random(n_lhs)) / n_lhs
                rows.append(lo + span * u[:, None])
        # fill with uniform
        while len(rows) < init:
            rows.append(lo + span * rng.random(dim))
        X = np.array(rows[:init], dtype=float)

        F = np.array([problem(x) for x in X], dtype=float)
        used = init
        Fok = valid(F)
        F = np.where(Fok[:, None], F, ref)

        # ---------------- vectorized helpers ----------------
        def hv_contrib(points):
            pts = np.asarray(points, dtype=float)
            if len(pts) == 0:
                return 0.0
            pts = pts[np.all(pts < ref, axis=1)]
            if len(pts) == 0:
                return 0.0
            order = np.lexsort(pts[:, ::-1])
            pts = pts[order]
            best = np.empty(n_obj, dtype=float)
            best.fill(np.inf)
            vol = 0.0
            prev = ref.copy()
            for p in pts:
                for k in range(n_obj):
                    if p[k] < best[k]:
                        seg = 1.0
                        for j in range(k):
                            seg *= (best[j] - p[j])
                        vol += seg * (prev[k] - p[k])
                        best[k] = p[k]
                prev = p
            return float(vol)

        def nds_sort(f):
            m = len(f)
            if m == 0:
                return []
            front = [0] * m
            count = [0] * m
            idx = list(range(m))
            while idx:
                cur = []
                for i in idx:
                    dom = False
                    for j in idx:
                        if i == j:
                            continue
                        if np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                            dom = True
                            break
                    if dom:
                        count[i] += 1
                    else:
                        cur.append(i)
                if not cur:
                    break
                for i in cur:
                    front[i] = 0
                for i in cur:
                    for j in idx:
                        if i == j:
                            continue
                        if np.all(f[i] <= f[j]) and np.any(f[i] < f[j]):
                            count[j] -= 1
                            if count[j] == 0:
                                front[j] = 1
                idx = [i for i in idx if front[i] == 0]
            fronts = []
            maxf = max(front) if front else 0
            for lv in range(maxf + 1):
                fronts.append([i for i in range(m) if front[i] == lv])
            return fronts

        def crowd_select(front, f, keep):
            if len(front) <= keep:
                return list(front)
            d = np.zeros(len(front), dtype=float)
            for k in range(n_obj):
                idx = np.argsort(f[front, k])
                order = [front[i] for i in idx]
                d[order[0]] = np.inf
                d[order[-1]] = np.inf
                rngk = f[front, k].max() - f[front, k].min()
                if rngk <= 1e-12:
                    continue
                for t in range(1, len(order) - 1):
                    d[order[t]] += (f[front, k][idx[t + 1]] - f[front, k][idx[t - 1]]) / rngk
            keep_idx = np.argsort(d)[::-1][:keep]
            return [front[i] for i in keep_idx]

        def hv_select(front, f, keep):
            if len(front) <= keep:
                return list(front)
            base = hv_contrib(f[front])
            scored = []
            for i in front:
                rest = [j for j in front if j != i]
                scored.append((base - hv_contrib(f[rest]), i))
            scored.sort(reverse=True)
            return [i for _, i in scored[:keep]]

        # ---------------- main loop ----------------
        while used < budget:
            if time.time() > deadline:
                break
            fronts = nds_sort(F)
            # hypervolume-guided archive: keep a spread subset of the best fronts
            keep = min(len(F), 24)
            archive = []
            for fr in fronts:
                if len(archive) + len(fr) <= keep:
                    archive.extend(fr)
                else:
                    archive.extend(hv_select(fr, F, keep - len(archive)))
                    break
            # mutation size adapts to progress
            base = 0.22
            frac = 1.0 - (used / float(budget))
            sigma = base * (0.35 + 0.85 * frac)
            n_child = 6
            for _ in range(n_child):
                if used >= budget or time.time() > deadline:
                    break
                if len(archive) == 0:
                    p = X[rng.integers(len(X))]
                else:
                    p = X[archive[rng.integers(len(archive))]]
                child = p + sigma * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = np.asarray(problem(child), dtype=float)
                used += 1
                if not np.all(np.isfinite(fc)):
                    continue
                # replace the member with the lowest hypervolume contribution
                if len(archive) < 2:
                    j = rng.integers(len(X))
                else:
                    base = hv_contrib(F[archive])
                    contrib = np.array([base - hv_contrib(np.delete(F[archive], t))
                                        for t in range(len(archive))])
                    j = archive[int(np.argmin(contrib))]
                X[j] = child
                F[j] = fc
        return X
# EVOLVE-BLOCK-END
