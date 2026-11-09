# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin well under 300s per eval

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        rng = np.random.default_rng(12345)
        used = 0

        def clip(x):
            return np.clip(x, lo, hi)

        def time_ok():
            return time.time() < deadline

        # --- archive of all recorded (x, f) ---
        Xs = []
        Fs = []

        def add(x, f):
            nonlocal used
            Xs.append(np.asarray(x, dtype=float))
            Fs.append(np.asarray(f, dtype=float))
            used += 1

        # --- non-domination ranks over a list of f-vectors ---
        def ranks(F):
            n = len(F)
            if n == 0:
                return []
            F = np.asarray(F)
            ranks = np.zeros(n, dtype=int)
            assigned = np.zeros(n, dtype=bool)
            front = 0
            while np.any(~assigned):
                un = np.where(~assigned)[0]
                dom = np.zeros(len(un), dtype=bool)
                for i, a in enumerate(un):
                    fa = F[a]
                    # is a dominated by any unassigned point b?
                    for j, b in enumerate(un):
                        if i == j:
                            continue
                        fb = F[b]
                        if np.all(fb <= fa) and np.any(fb < fa):
                            dom[i] = True
                            break
                front_idx = un[~dom]
                for idx in front_idx:
                    assigned[idx] = True
                    ranks[idx] = front
                front += 1
            return ranks.tolist()

        def nondominated_idx(F):
            n = len(F)
            if n == 0:
                return []
            F = np.asarray(F)
            keep = []
            for i in range(n):
                fi = F[i]
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    fj = F[j]
                    if np.all(fj <= fi) and np.any(fj < fi):
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)
            return keep

        def crowding(F):
            # returns density per point (larger = more crowded)
            n = len(F)
            if n <= 2:
                return np.zeros(n)
            F = np.asarray(F)
            dens = np.zeros(n)
            for o in range(n_obj):
                order = np.argsort(F[:, o])
                dens[order[0]] = np.inf
                dens[order[-1]] = np.inf
                rngv = F[order[-1], o] - F[order[0], o]
                if rngv <= 0:
                    continue
                for k in range(1, n - 1):
                    dens[order[k]] += (F[order[k + 1], o] - F[order[k - 1], o]) / rngv
            return dens

        # --- initial population ---
        pop = max(8, min(24, budget // 4))
        X = lo + span * rng.random((pop, dim))
        for x in X:
            add(x, problem(x))
        if not time_ok():
            return X

        F = np.asarray(Fs)

        # --- evolutionary loop ---
        while used < budget and time_ok():
            r = ranks(F)
            # pick parents from best front(s)
            best_front = min(r)
            cand = [i for i, ri in enumerate(r) if ri == best_front]
            if len(cand) < 2:
                cand = list(range(len(F)))
            # pick two parents (prefer less crowded within front for diversity)
            d = crowding(F[cand])
            # choose parents weighted toward low density
            p1 = cand[int(rng.integers(len(cand)))]
            p2 = cand[int(rng.integers(len(cand)))]
            if len(cand) > 2:
                # try to pick low-density parent
                p1 = cand[int(np.argmin(d))]
                p2 = cand[int(rng.integers(len(cand)))]

            # crossover + mutation
            a = F[p1]
            b = F[p2]
            # blend
            alpha = rng.random()
            child = a * alpha + b * (1 - alpha)
            # mutation
            mut = rng.random(dim) < 0.3
            child[mut] += 0.05 * span[mut] * rng.standard_normal(int(mut.sum()))
            child = clip(child)
            fc = problem(child)
            add(child, fc)
            F = np.asarray(Fs)

            # replace worst individual (highest rank, then highest sum)
            r = ranks(F)
            worst_front = max(r)
            worst_cand = [i for i, ri in enumerate(r) if ri == worst_front]
            sums = F[worst_cand].sum(axis=1)
            worst = worst_cand[int(np.argmax(sums))]
            Xs[worst] = np.asarray(child, dtype=float)
            Fs[worst] = np.asarray(fc, dtype=float)
            F = np.asarray(Fs)

        # --- memetic local search on best non-dominated points ---
        F = np.asarray(Fs)
        nd = nondominated_idx(F)
        if len(nd) == 0:
            nd = [int(np.argmin(F.sum(axis=1)))]
        # take a handful of the most promising non-dominated points
        # sort by sum (lower = closer to ideal) and take top few
        order = sorted(nd, key=lambda i: F[i].sum())
        top = order[:max(1, min(6, len(order)))]

        for i in top:
            if not time_ok():
                break
            x0 = Xs[i].copy()
            try:
                res = minimize(
                    lambda v: float(np.sum(problem(clip(v)))),
                    x0,
                    method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"maxiter": 200},
                )
                if time_ok():
                    add(res.x, problem(clip(res.x)))
            except Exception:
                pass
            # a second, finer local step from the result
            if time_ok():
                try:
                    res2 = minimize(
                        lambda v: float(np.sum(problem(clip(v)))),
                        res.x,
                        method="Nelder-Mead",
                        options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-9},
                    )
                    if time_ok():
                        add(res2.x, problem(clip(res2.x)))
                except Exception:
                    pass

        return Xs
# EVOLVE-BLOCK-END
