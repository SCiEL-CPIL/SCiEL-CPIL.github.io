"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before the 330 s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        N = 50
        cr = 0.9
        F1 = 0.5
        used = 0
        pop = 0
        F = None

        def fast_non_dominated(F):
            m = len(F)
            ranks = np.empty(m, dtype=int)
            fronts = []
            idx = np.arange(m)
            while len(idx):
                best = []
                for i in idx:
                    fi = F[i]
                    dom = (F[idx] <= fi).all(axis=1) & (F[idx] < fi).any(axis=1)
                    if not dom.any():
                        best.append(i)
                if not best:
                    best = [idx[0]]
                for i in best:
                    ranks[i] = len(fronts)
                fronts.append(best)
                idx = idx[~np.isin(idx, best)]
            return ranks, fronts

        def crowding(F, front):
            n = len(front)
            cd = np.ones(n)
            if n <= 2:
                return cd
            for k in range(F.shape[1]):
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng = F[order[-1], k] - F[order[0], k]
                if rng > 0:
                    for j in range(1, n - 1):
                        cd[order[j]] += (F[order[j + 1], k] - F[order[j - 1], k]) / rng
            return cd

        def select_parents(ranks, F):
            fronts = fast_non_dominated(F)[1]
            pool = []
            for fr in fronts:
                if len(pool) + len(fr) <= N:
                    pool.extend(fr)
                else:
                    need = N - len(pool)
                    cd = crowding(F, fr)
                    sel = np.argsort(-cd)[:need]
                    pool.extend([fr[i] for i in sel])
                    break
            return pool

        def tournament(pool, k=2):
            c = np.random.choice(pool, size=k, replace=False)
            return int(c[np.argmin(F[c].sum(axis=1))])

        def crossover(a, b):
            if np.random.rand() > cr:
                return a.copy(), b.copy()
            c1 = a.copy()
            c2 = b.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    c1[i], c2[i] = b[i], a[i]
            return c1, c2

        def mutate(x):
            m = np.random.random(self.dim) < 0.1
            x = x.copy()
            x[m] = lo[m] + np.random.random(m.sum()) * span[m]
            return x

        # initial population
        X = lo + np.random.random((N, self.dim)) * span
        F = np.array([problem(x) for x in X])
        used = N
        while used < self.budget and time.time() < deadline:
            ranks, fronts = fast_non_dominated(F)
            pool = select_parents(ranks, F)
            children = []
            for _ in range(N):
                p1 = tournament(pool)
                p2 = tournament(pool)
                c1, c2 = crossover(X[p1], X[p2])
                children.append(mutate(c1))
                children.append(mutate(c2))
            children = np.array(children[:N])
            Fc = np.array([problem(x) for x in children])
            used += N
            R = np.vstack([X, children])
            FR = np.vstack([F, Fc])
            rr, frs = fast_non_dominated(FR)
            newpool = []
            for fr in frs:
                if len(newpool) + len(fr) <= N:
                    newpool.extend(fr)
                else:
                    need = N - len(newpool)
                    cd = crowding(FR, fr)
                    sel = np.argsort(-cd)[:need]
                    newpool.extend([fr[i] for i in sel])
                    break
            X = R[newpool]
            F = FR[newpool]
        return X
# EVOLVE-BLOCK-END
