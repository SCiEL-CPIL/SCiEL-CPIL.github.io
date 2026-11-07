# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Hybrid multi-objective optimizer: Sobol init + multi-start weighted-sum L-BFGS-B
    local search + NSGA-II-style non-dominated/crowding refinement, under a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin well under 300 s

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (hi + lo)
        rng = np.random.default_rng(0)

        evals = [0]

        def call(x):
            evals[0] += 1
            y = problem(x)
            y = np.asarray(y, dtype=float)
            return y

        # ---- Sobol quasi-random initialization ----
        n_init = min(64, budget)
        if n_init >= 2:
            sampler = qmc.Sobol(d=dim, scramble=True)
            init = sampler.random(n_init)
        else:
            init = rng.random((n_init, dim))
        X = lo + span * init
        F = np.array([call(x) for x in X])

        # ---- Multi-start weighted-sum local search ----
        # Build a spread of linear weight vectors over the simplex.
        n_ws = min(40, max(8, budget // 8))
        # Use Dirichlet to sample weights, plus the axis vertices.
        wts = [np.zeros(n_obj) for _ in range(n_obj)]
        for k in range(n_obj):
            wts[k][k] = 1.0
        extra = max(0, n_ws - n_obj)
        if extra > 0:
            wts.append(np.array([1.0] * n_obj) / n_obj)
            extra -= 1
        if extra > 0:
            w_extra = rng.dirichlet(np.ones(n_obj), size=extra)
            wts.extend(w_extra.tolist())
        wts = np.array(wts, dtype=float)

        # For each weight, run L-BFGS-B from a few random starts (and a midpoint start).
        n_starts_per_w = 2
        for w in wts:
            if evals[0] >= budget:
                break
            starts = [mid.copy()]
            for _ in range(n_starts_per_w):
                if evals[0] >= budget:
                    break
                starts.append(lo + span * rng.random(dim))
            for x0 in starts:
                if evals[0] >= budget:
                    break
                if time.time() > deadline - 5.0:
                    return X

                def obj(x):
                    return float(np.dot(w, call(x)))

                try:
                    res = minimize(obj, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 250, 'maxfun': 200,
                                            'maxls': 250})
                    if res.x is not None:
                        xf = np.clip(res.x, lo, hi)
                        Ff = call(xf)
                        X = np.vstack([X, xf])
                        F = np.vstack([F, Ff])
                except Exception:
                    pass

        # ---- NSGA-II-style refinement loop ----
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [-1] * n
            fronts = []
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
                    fronts.append([p])
            i = 0
            while i < len(fronts):
                current = fronts[i]
                nxt = []
                for p in current:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            rank[q] = i + 1
                            nxt.append(q)
                if nxt:
                    fronts.append(nxt)
                i += 1
            return rank

        def crowding(F, idx):
            n = len(idx)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            Fsub = F[idx]
            for m in range(n_obj):
                order = np.argsort(Fsub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rngm = Fsub[order[-1], m] - Fsub[order[0], m]
                if rngm <= 1e-12:
                    continue
                for j in range(1, n - 1):
                    cd[order[j]] += (Fsub[order[j + 1], m] - Fsub[order[j - 1], m]) / rngm
            return cd

        # Refinement: keep best ~pop_size by (rank, crowding), mutate non-dominated,
        # evaluate children, merge, re-sort.
        pop_size = 24
        while evals[0] < budget and time.time() < deadline - 2.0:
            if len(F) < 2:
                break
            rank = non_dominated_sort(F)
            # Build fronts in order
            max_rank = max(rank) + 1 if len(rank) else 0
            fronts = [[] for _ in range(max_rank)]
            for i, r in enumerate(rank):
                fronts[r].append(i)
            # Truncate to pop_size
            keep = []
            for fr in fronts:
                if len(keep) + len(fr) <= pop_size:
                    keep.extend(fr)
                else:
                    need = pop_size - len(keep)
                    if need > 0:
                        cd = crowding(F, fr)
                        order = np.argsort(cd)[::-1]
                        keep.extend([fr[i] for i in order[:need]])
                    break
            F = F[keep]
            X = X[keep]
            # Select parents from first front (non-dominated)
            nd_idx = [i for i, r in enumerate(rank) if r == 0 and i in keep]
            if not nd_idx:
                nd_idx = list(range(len(F)))
            # Mutate a few children
            n_child = min(8, pop_size // 3, budget - evals[0])
            if n_child <= 0:
                break
            new_X = []
            new_F = []
            for _ in range(n_child):
                if evals[0] >= budget:
                    break
                p = F[nd_idx[rng.integers(len(nd_idx))]]
                x = X[nd_idx[rng.integers(len(nd_idx))]]
                sigma = 0.15 * span
                x = np.clip(x + sigma * rng.standard_normal(dim), lo, hi)
                fx = call(x)
                new_X.append(x)
                new_F.append(fx)
            if new_X:
                X = np.vstack([X, np.array(new_X)])
                F = np.vstack([F, np.array(new_F)])
        return X
# EVOLVE-BLOCK-END
