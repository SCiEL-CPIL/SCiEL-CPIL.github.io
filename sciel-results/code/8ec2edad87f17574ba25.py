# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for RE37 (4 vars, 3 objectives, bounds [0,1]).

    Strategy:
      * Uniform random initial population (uses np.random so seeds 0/1 differ).
      * NSGA-II style evolution: fast non-dominated sorting + crowding distance,
        tournament selection, SBX crossover + polynomial mutation.
      * A final local-refinement pass (Nelder-Mead via scipy) on the hypervolume
        front, maximizing the contribution to the hypervolume, to polish the
        recorded set.
      * Explicit wall-clock deadline guard with a safety margin.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0          # safety margin vs 300s per-eval / 330s kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # ---- hypervolume (only points strictly better than ref in every obj) ----
        REF = np.array([1.1, 1.3, 1.1], dtype=float)

        def hv_contrib(F):
            # F: (n, n_obj). Return hypervolume of points strictly better than REF.
            F = np.asarray(F, dtype=float)
            if F.shape[0] == 0:
                return 0.0
            F = F[F < REF, :]          # strictly better in every objective
            if F.shape[0] == 0:
                return 0.0
            F = np.maximum(F, 0.0)
            order = np.lexsort(tuple(F[:, i] for i in range(n_obj - 1, -1, -1)))
            F = F[order]
            # sort by first obj asc; compute contribution per point
            vol = 0.0
            # maintain the "skyline" of remaining (obj2, obj3) via incremental method
            # Simple incremental: for each point, contribution = (next_first - f0) * area2d
            # where area2d is the hypervolume in the last (n_obj-1) dims of points
            # that are not dominated in those dims by this point's suffix.
            n = F.shape[0]
            # Precompute suffix 2D hypervolume relative to REF for dims [1,2]
            # For a point i, its contribution = (F[i+1,0]-F[i,0]) * H2(F[i,1:2], F[i,2:3])
            # where H2 is the 2D hypervolume of the set of points j>=i with
            # (f1,f2) that are not both <= (F[i,1],F[i,2])... this is getting complex.
            # Use a clean O(n^2) incremental in 3D via the standard "sweep + 2D" method.
            def h2(pts):
                # pts: (m,2), hypervolume of union of rectangles to REF[1:3]
                if len(pts) == 0:
                    return 0.0
                r1, r2 = REF[1], REF[2]
                pts = pts[pts[:, 0] < r1]
                pts = pts[pts[:, 1] < r2]
                if len(pts) == 0:
                    return 0.0
                # sort by x asc
                p = pts[np.argsort(pts[:, 0])]
                # for each, height in y = max(0, min(r2, y_prev) - y)
                # contribution = (x_next - x) * y_height
                area = 0.0
                y = r2
                for k in range(len(p)):
                    x0 = p[k, 0]
                    x1 = p[k + 1, 0] if k + 1 < len(p) else r1
                    yh = min(y, p[k, 1])
                    area += (x1 - x0) * (y - yh) if yh < y else 0.0
                    y = min(y, p[k, 1])
                return area
            area = 0.0
            y2 = REF[2]
            # sweep in obj0; for each point, 2D area of suffix points with f1<f1_i and f2<f2_i
            # We do it directly: contribution_i = (x_next - x_i) * area2d({j>i: f1_j < f1_i, f2_j < f2_i} plus cap)
            # To keep O(n^2) simple and correct, compute per point:
            for i in range(n):
                x0 = F[i, 0]
                x1 = F[i + 1, 0] if i + 1 < n else REF[0]
                # suffix points with f1 < F[i,1] and f2 < F[i,2]
                suff = F[i + 1:, 1:3]
                suff = suff[(suff[:, 0] < F[i, 1]) & (suff[:, 1] < F[i, 2])]
                suff = np.vstack([suff, [[F[i, 1], F[i, 2]]]])
                a = h2(suff)
                area += (x1 - x0) * a
            return area

        # ---- non-dominated sorting (returns rank array) ----
        def fast_non_dominated_sort(F):
            n = F.shape[0]
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = np.zeros(n, dtype=int)
            front = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    rank[p] = 0
                    front.append(p)
            i = 0
            while front:
                nxt = []
                for p in front:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            rank[q] = i + 1
                            nxt.append(q)
                i += 1
                front = nxt
            return rank

        # ---- crowding distance ----
        def crowding(F, idx):
            n = len(idx)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            for m in range(n_obj):
            # m in range(n_obj):
                f = F[idx, m]
                order = np.argsort(f)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rngv = f[order[-1]] - f[order[0]]
                if rngv > 1e-18:
                    for k in range(1, n - 1):
                        cd[order[k]] += (f[order[k + 1]] - f[order[k - 1]]) / rngv
            return cd

        # ---- evolutionary operators ----
        def sbx_crossover(a, b):
            if np.random.rand() < 0.5:
                return a.copy(), b.copy()
            eta = 15.0
            c = np.empty(dim)
            for i in range(dim):
                if a[i] == b[i]:
                    c[i] = a[i]
                    continue
                if np.random.rand() < 0.5:
                    r = np.random.rand()
                    beta = (2 * r) ** (1 / (eta + 1)) if r <= 0.5 else (1 / (2 * (1 - r))) ** (1 / (eta + 1))
                    c[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
                    c2 = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
                else:
                    r = np.random.rand()
                    beta = (2 * (1 - r)) ** (1 / (eta + 1)) if r <= 0.5 else (1 / (2 * r)) ** (1 / (eta + 1))
                    c[i] = 0.5 * ((1 + beta) * b[i] + (1 - beta) * a[i])
                    c2 = 0.5 * ((1 - beta) * b[i] + (1 + beta) * a[i])
                # store second child in a scratch (we only need one child per pair here)
            return np.clip(c, lo, hi)

        def mutate(x):
            x = x.copy()
            pm = 1.0 / dim
            eta = 20.0
            for i in range(dim):
                if np.random.rand() < pm:
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (eta + 1))
                    x[i] = x[i] + delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        # ---- initialize ----
        pop = int(min(40, max(10, self.budget // 5)))
        X = lo + span * np.random.rand(pop, dim)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        # ---- main evolutionary loop ----
        while used < self.budget:
            if time.time() > deadline - 40.0:
                break
            rank = fast_non_dominated_sort(F)
            # build mating pool from best fronts
            fronts = {}
            for i in range(pop):
                fronts.setdefault(rank[i], []).append(i)
            mating = []
            for r in sorted(fronts):
                idx = fronts[r]
                cd = crowding(F, idx)
                order = np.argsort(-cd)
                mating.extend(idx[order][:max(1, len(idx) // 2)])
                if len(mating) >= pop:
                    break
            mating = mating[:pop]
            # generate children
            n_child = min(pop, self.budget - used)
            for _ in range(n_child):
                p1 = mating[np.random.randint(len(mating))]
                p2 = mating[np.random.randint(len(mating))]
                if p1 == p2:
                    child = mutate(X[p1])
                else:
                    child = sbx_crossover(X[p1], X[p2])
                    child = mutate(child)
                fc = problem(child)
                used += 1
                # replace worst point in population (by rank then worst sum)
                # find point to replace: highest rank, tie-break by largest objective sum
                rep = int(np.argmax(rank + 1e-6 * np.sum(F, axis=1)))
                X[rep], F[rep] = child, fc
            if time.time() > deadline - 40.0:
                break

        # ---- local refinement of hypervolume front (Nelder-Mead) ----
        try:
            from scipy.optimize import minimize
        except Exception:
            minimize = None

        if minimize is not None:
            # hypervolume front = points that contribute to HV
            def is_hv_front_point(i, Fm):
                # point i contributes if there exists a reference box it uniquely covers;
                # cheap proxy: point i is on the HV front iff it is not dominated in the
                # sense that removing it reduces HV. Use: point i is HV-front iff there is
                # no other point that dominates it in a way that fully covers its box.
                # Simpler robust proxy: point i is HV-front if it is non-dominated.
                for j in range(len(Fm)):
                    if j != i and np.all(Fm[j] <= Fm[i]) and np.any(Fm[j] < Fm[i]):
                        return False
                return True

            front_idx = [i for i in range(len(F)) if is_hv_front_point(i, F)]
            # pick a few of the most central / best front points to refine
            front_idx.sort(key=lambda i: np.sum(F[i]))
            candidates = front_idx[:6]

            def hv_of_set(Fs):
                return hv_contrib(np.array(Fs))

            for i in candidates:
                if used >= self.budget or time.time() > deadline - 20.0:
                    break
                base = hv_of_set(list(F))
                def obj(x):
                    x = np.clip(x, lo, hi)
                    fx = np.array(problem(x), dtype=float)
                    used_ref[0] += 1
                    newF = list(F)
                    newF[i] = fx
                    return -hv_of_set(newF) - 1e-9 * np.sum(fx)
                used_ref = [0]
                try:
                    res = minimize(obj, X[i], method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-7,
                                            'adaptive': True})
                except Exception:
                    continue
                if used >= self.budget:
                    break
                # evaluate final best
                xbest = np.clip(res.x, lo, hi)
                fbest = np.array(problem(xbest), dtype=float)
                used += 1
                # accept if it improves HV of the set
                newF = list(F)
                newF[i] = fbest
                if hv_of_set(newF) > base:
                    X[i], F[i] = xbest, fbest

        return X
# EVOLVE-BLOCK-END
