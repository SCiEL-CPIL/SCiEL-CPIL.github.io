"""Multi-objective NSGA-II-style optimizer on ZDT3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style search: tournament + crowding selection, hypervolume archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s wall-clock kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        def clip(x):
            return np.clip(x, lo, hi)

        def fast_nondominated(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            ifort = np.argsort(F[:, 0])
            for i in ifort:
                Fi = F[i]
                for j in range(n):
                    if i == j:
                        continue
                    Fj = F[j]
                    if np.all(Fj <= Fi) and np.any(Fj < Fi):
                        S[j].append(i)
                        nd[i] += 1
            frontier = np.where(nd == 0)[0]
            ifort = list(frontier)
            while ifort:
                nxt = []
                for i in ifort:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            nxt.append(j)
                frontier = np.array(nxt)
                ifort = list(frontier)
            return ifort

        def crowding(F):
            n = len(F)
            if n <= 2:
                return np.full(n, 1e30)
            cd = np.zeros(n)
            for m in range(n_obj):
                order = np.argsort(F[:, m])
                cd[order[0]] = 1e30
                cd[order[-1]] = 1e30
                rng = F[order[-1], m] - F[order[0], m]
                if rng > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / rng
            return cd

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- population init ---
        pop = 100
        X = lo + span * np.random.rand(pop, dim)
        F = np.array([problem(x) for x in X])
        used = pop
        archive = list(zip(X.tolist(), F.tolist()))

        # --- main loop ---
        while used < self.budget:
            if time.time() >= deadline:
                break

            # ranks
            ranks = np.zeros(len(F), dtype=int)
            ifort = fast_nondominated(F)
            rem = set(range(len(F)))
            r = 0
            while ifort and rem:
                for i in ifort:
                    ranks[i] = r
                    rem.discard(i)
                r += 1
                if not rem:
                    break
                ifort = fast_nondominated(F[list(rem)])
            for i in rem:
                ranks[i] = r

            # crowding per rank
            cd = np.zeros(len(F))
            for i in range(len(F)):
                idx = np.where(ranks == ranks[i])[0]
                if len(idx) > 1:
                    cd[i] = crowding(F[idx])[list(idx).index(i)]

            # tournament select 2 parents
            def select():
                a, b = np.random.randint(0, len(F), 2)
                if ranks[a] < ranks[b]:
                    return a
                if ranks[b] < ranks[a]:
                    return b
                return a if cd[a] > cd[b] else b

            p1, p2 = select(), select()
            xp1, xp2 = X[p1], X[p2]

            # polynomial crossover
            if np.random.rand() < 0.9:
                eta = 20.0
                u = np.random.rand(dim)
                c1 = np.zeros(dim); c2 = np.zeros(dim)
                for k in range(dim):
                    if xp1[k] == xp2[k]:
                        c1[k] = c2[k] = xp1[k]
                        continue
                    if u[k] <= 0.5:
                        beta = (2.0 * u[k]) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u[k]))) ** (1.0 / (eta + 1.0))
                    c1[k] = 0.5 * ((1.0 + beta) * xp1[k] + (1.0 - beta) * xp2[k])
                    c2[k] = 0.5 * ((1.0 + 1.0 / beta) * xp1[k] + (1.0 - 1.0 / beta) * xp2[k])
                child1, child2 = clip(c1), clip(c2)
            else:
                child1, child2 = clip(xp1.copy()), clip(xp2.copy())

            # binary mutation
            pm = 1.0 / dim
            for child in (child1, child2):
                mask = np.random.rand(dim) < pm
                if np.any(mask):
                    child[mask] = lo[mask] + span[mask] * np.random.rand(np.count_nonzero(mask))

            # evaluate child1
            f1 = np.array(problem(child1))
            used += 1
            archive.append((child1.tolist(), f1.tolist()))
            if dominates(f1, F[p1]):
                X[p1], F[p1] = child1, f1
            elif not dominates(F[p1], f1):
                X[np.random.randint(len(F))], F[np.random.randint(len(F))] = child1, f1

            if used >= self.budget or time.time() >= deadline:
                break

            # evaluate child2
            f2 = np.array(problem(child2))
            used += 1
            archive.append((child2.tolist(), f2.tolist()))
            if dominates(f2, F[p2]):
                X[p2], F[p2] = child2, f2
            elif not dominates(F[p2], f2):
                X[np.random.randint(len(F))], F[np.random.randint(len(F))] = child2, f2

        return X

    @staticmethod
    def _hypervolume_2d(pts, ref):
        # pts: (n,2) strictly better than ref; returns 2D hypervolume
        if len(pts) == 0:
            return 0.0
        pts = sorted(pts.tolist(), key=lambda p: p[0])
        hv = 0.0
        prev_y = ref[1]
        for x, y in pts:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv
# EVOLVE-BLOCK-END
