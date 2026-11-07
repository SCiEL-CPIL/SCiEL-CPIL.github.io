"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fast_non_dominated_sort(F):
    """Return list of fronts (each a list of indices), best first."""
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts = []
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
                nd[p] += 1
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                nd[q] += 1
    front0 = [i for i in range(n) if nd[i] == 0]
    fronts.append(front0)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
    return fronts


def _crowding_distance(F, idx):
    """Crowding distance for indices idx (list of ints)."""
    m = len(idx)
    if m <= 2:
        return np.full(m, np.inf)
    cd = np.zeros(m)
    for k in range(F.shape[1]):
        vals = F[idx, k]
        order = np.argsort(vals)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        rng = vals[order[-1]] - vals[order[0]]
        if rng > 0:
            for j in range(1, m - 1):
                cd[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / rng
    return cd


class Optimizer:
    """Multi-objective optimizer on ZDT6: NSGA-II with structure-aware seeding + polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before the 330s wall-clock limit
        dim = self.dim
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        def ev(x):
            return np.asarray(problem(x), dtype=float)

        # ---- Structure-aware seed: spread x1 (drives f1), random rest (drives f2) ----
        n_seed = 12
        X = np.empty((n_seed, dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_seed)
        X[:, 1:] = rng.random((n_seed, dim - 1))
        F = np.array([ev(x) for x in X])
        used = n_seed

        # ---- NSGA-II ----
        N = 40
        # top-up population with random individuals
        if n_seed < N:
            extra = N - n_seed
            Xr = lo + (hi - lo) * rng.random((extra, dim))
            Fr = np.array([ev(x) for x in Xr])
            used += extra
            X = np.vstack([X, Xr])
            F = np.vstack([F, Fr])

        def select_parents(popF, popX):
            fronts = _fast_non_dominated_sort(popF)
            sel = []
            for fr in fronts:
                if len(sel) + len(fr) <= N:
                    sel.extend(fr)
                else:
                    need = N - len(sel)
                    cd = _crowding_distance(popF, fr)
                    order = np.argsort(-cd)
                    sel.extend([fr[i] for i in order[:need]])
                    break
            return [popX[i] for i in sel]

        while used < self.budget:
            if time.time() > deadline - 25.0:
                break
            parents = select_parents(F, X)
            # generate up to N offspring (respect budget)
            n_off = min(N, self.budget - used)
            children = []
            for _ in range(n_off):
                i = rng.integers(len(parents))
                j = rng.integers(len(parents))
                if i == j:
                    j = (j + 1) % len(parents)
                a, b = parents[i], parents[j]
                # blend + mutation
                child = a + rng.random() * (b - a)
                if rng.random() < 0.5:
                    child += 0.05 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                children.append(child)
            C = np.array(children)
            CF = np.array([ev(x) for x in C])
            used += n_off
            # combine and select N
            RX = np.vstack([X, C])
            RF = np.vstack([F, CF])
            fronts = _fast_non_dominated_sort(RF)
            keep = []
            for fr in fronts:
                if len(keep) + len(fr) <= N:
                    keep.extend(fr)
                else:
                    need = N - len(keep)
                    cd = _crowding_distance(RF, fr)
                    order = np.argsort(-cd)
                    keep.extend([fr[i] for i in order[:need]])
                    break
            X = RX[keep]
            F = RF[keep]

        # ---- Final polish: local search around best points for f2 with x1 fixed ----
        if time.time() < deadline - 5.0:
            fronts = _fast_non_dominated_sort(F)
            best_idx = list(fronts[0])[:min(8, len(fronts[0]))]
            for bi in best_idx:
                if time.time() > deadline - 3.0:
                    break
                x0 = X[bi].copy()
                f0 = F[bi].copy()
                step = 0.02
                for _ in range(40):
                    if time.time() > deadline - 2.0:
                        break
                    improved = False
                    for d in range(dim):
                        for sgn in (1.0, -1.0):
                            cand = x0.copy()
                            cand[d] += sgn * step
                            cand = np.clip(cand, lo, hi)
                            fc = ev(cand)
                            # accept if it dominates or improves f2 while keeping f1
                            if np.all(fc <= f0) and np.any(fc < f0):
                                x0, f0 = cand, fc
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break
                # record the polished point
                X = np.vstack([X, x0])
                F = np.vstack([F, f0])
                used += 1
                if used > self.budget:
                    break

        return X
# EVOLVE-BLOCK-END
