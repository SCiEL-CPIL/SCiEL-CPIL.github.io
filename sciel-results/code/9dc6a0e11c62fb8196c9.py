"""Multi-objective NSGA-II optimizer for WFG7 with a strict wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 12.0  # safety margin before the 300s kill

        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random.default_rng(0)

        # ---- dominance helpers (vectorized) ----
        def dominates(A, B):
            # A, B: (n, n_obj)  -> bool (n): A[i] dominates B[i]
            le = np.all(A <= B, axis=1)
            lt = np.any(A < B, axis=1)
            return le & lt

        def fast_nondominated_sort(F):
            n = F.shape[0]
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            cur = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        nd[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    cur.append(p)
            while cur:
                fronts.append(np.array(cur, dtype=int))
                nxt = []
                for p in cur:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                cur = nxt
            return fronts

        def crowding(F, idx):
            n = len(idx)
            if n <= 2:
                return np.full(n, np.inf)
            Ff = F[idx]
            cd = np.zeros(n)
            for m in range(n_obj):
            # guard against zero range
                mn, mx = Ff[:, m].min(), Ff[:, m].max()
                if mx - mn < 1e-30:
                    continue
                order = np.argsort(Ff[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (Ff[order[k + 1], m] - Ff[order[k - 1], m]) / (mx - mn)
            return cd

        # ---- population ----
        pop = int(min(240, max(20, budget // 4)))
        X = lo + span * rng.random((pop, dim))
        F = np.asarray(problem(X), dtype=float)
        used = pop

        # ---- evolution loop ----
        while used < budget:
            if time.time() >= deadline:
                break
            fronts = fast_nondominated_sort(F)
            # rank per individual
            rank = np.empty(F.shape[0], dtype=int)
            for r, fr in enumerate(fronts):
                rank[fr] = r

            # build mating pool: fill with fronts, use crowding to break ties on last front
            pool = []
            for fr in fronts:
                if len(pool) + len(fr) <= pop:
                    pool.extend(fr.tolist())
                else:
                    need = pop - len(pool)
                    cd = crowding(F, fr)
                    order = np.argsort(-cd)
                    pool.extend(fr[order[:need]].tolist())
                    break
            pool = np.array(pool, dtype=int)

            # generate children (up to one generation, bounded by remaining budget)
            room = min(pop, budget - used)
            if room <= 0:
                break
            # binary tournament
            def select():
                a = rng.integers(pop)
                b = rng.integers(pop)
                ra, rb = rank[a], rank[b]
                if ra != rb:
                    return a if ra < rb else b
                # same rank -> prefer lower crowding index in pool? use position
                ia = int(np.where(pool == a)[0][0]) if a in pool else 0
                ib = int(np.where(pool == b)[0][0]) if b in pool else 0
                return a if ia <= ib else b

            children_X = np.empty((room, dim))
            for i in range(room):
                p1 = select()
                p2 = select()
                # SBX-style blend with Gaussian mutation
                alpha = 20.0
                if rng.random() < 0.5:
                    beta = ((2 * rng.random()) ** (1 / (alpha + 1)))
                else:
                    beta = (1 / (2 * rng.random())) ** (1 / (alpha + 1))
                    beta = 1 / beta
                c1 = 0.5 * ((1 + beta) * X[p1] + (1 - beta) * X[p2])
                c1 = c1 + 0.05 * span * rng.standard_normal(dim)
                children_X[i] = np.clip(c1, lo, hi)

            Fc = np.asarray(problem(children_X), dtype=float)
            used += room

            # combine and keep best pop via ranks + crowding
            Xc = np.vstack([X, children_X])
            Fc_all = np.vstack([F, Fc])
            fronts2 = fast_nondominated_sort(Fc_all)
            keep = []
            for fr in fronts2:
                if len(keep) + len(fr) <= pop:
                    keep.extend(fr.tolist())
                else:
                    need = pop - len(keep)
                    cd = crowding(Fc_all, fr)
                    order = np.argsort(-cd)
                    keep.extend(fr[order[:need]].tolist())
                    break
            keep = np.array(keep, dtype=int)
            X = Xc[keep]
            F = Fc_all[keep]

        return X
# EVOLVE-BLOCK-END
