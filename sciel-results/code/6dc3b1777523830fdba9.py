"""Rewritten multi-objective optimizer for WFG7 using NSGA-II."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective search with time guard and restarts."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _dominates(A, B):
        """A (n,m) dominates B (n,m) if A<=B everywhere and < somewhere."""
        le = np.all(A <= B, axis=1)
        lt = np.any(A < B, axis=1)
        return le & lt

    def _fast_non_dominated(self, F):
        """Return (fronts, ranks). fronts: list of index arrays, front 0 = best."""
        n = len(F)
        S = [[] for _ in range(n)]   # S[i] = indices dominated by i
        nd = [0] * n                 # number of solutions dominating i
        fronts = [[]]
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j])[0]:
                    S[i].append(j); nd[j] += 1
                elif self._dominates(F[j], F[i])[0]:
                    S[j].append(i); nd[i] += 1
            if nd[i] == 0:
                fronts[0].append(i)
        ranks = np.zeros(n, dtype=int)
        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                ranks[i] = k
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        ranks[j] = k + 1
                        nxt.append(j)
            k += 1
            fronts.append(nxt)
        fronts.pop()  # drop last empty
        return fronts, ranks

    def _crowding(self, F, idx):
        """Crowding distance for indices idx (list) in objective space F."""
        d = np.zeros(len(idx))
        m = self.n_obj
        for k in range(m):
            fk = F[idx, k]
            order = np.argsort(fk)
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            rng = fk[order[-1]] - fk[order[0]]
            if rng > 1e-12:
                for t in range(1, len(idx) - 1):
                    d[order[t]] += (fk[order[t + 1]] - fk[order[t - 1]]) / rng
        return d

    def _select_parents(self, X, F, ranks, n_sel):
        """Binary tournament on (rank, crowding)."""
        # crowding per front
        fronts, _ = self._fast_non_dominated(F)
        crowd = np.zeros(len(F))
        for fr in fronts:
            cd = self._crowding(F, fr)
            for i, v in zip(fr, cd):
                crowd[i] = v

        def fitness(i):
            return (ranks[i], -crowd[i])

        def tournament():
            a = np.random.randint(len(F)); b = np.random.randint(len(F))
            fa, fb = fitness(a), fitness(b)
            return a if (fa[0] < fb[0] or (fa[0] == fb[0] and fa[1] > fb[1])) else b

        return [tournament() for _ in range(n_sel)]

    def _sbx_crossover(self, p1, p2, lo, hi):
        if np.random.rand() < 0.9:
            beta = 1.0 / 0.5  # eta_c=20
            x = np.empty(self.dim)
            for d in range(self.dim):
                u = np.random.rand()
                if u <= 2 / (beta + 1):
                    c1 = (0.5 * (p1[d] + p2[d]) + 0.5 * (p1[d] - p2[d]) * (2 * u) ** (1 / (beta + 1)))
                    c2 = (0.5 * (p1[d] + p2[d]) - 0.5 * (p1[d] - p2[d]) * (2 * u) ** (1 / (beta + 1)))
                elif u >= 1 - 2 / (beta + 1):
                    c1 = (0.5 * (p1[d] + p2[d]) + 0.5 * (p1[d] - p2[d]) * (2 * (1 - u)) ** (1 / (beta + 1)))
                    c2 = (0.5 * (p1[d] + p2[d]) - 0.5 * (p1[d] - p2[d]) * (2 * (1 - u)) ** (1 / (beta + 1)))
                else:
                    c1, c2 = p1[d], p2[d]
                x[d] = c1
            return np.clip(x, lo, hi)
        return p1.copy()

    # ---- main ----------------------------------------------------------
    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 250.0   # safety margin below 300s / 330s kill
        rng = np.random.RandomState(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        N = 24
        used = 0

        def time_left():
            return (deadline - time.time()) > 1.0

        # initial population
        X = lo + span * rng.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used += N
        best = F.copy()

        stagnation = 0
        prev_best_count = -1

        while used < self.budget and time_left():
            fronts, ranks = self._fast_non_dominated(F)
            # record improvement of best front
            cur_best_count = len(fronts[0])
            if cur_best_count > prev_best_count:
                stagnation = 0
            else:
                stagnation += 1
            prev_best_count = cur_best_count

            # restart if long stagnation
            if stagnation > 12 and used + N < self.budget:
                X = lo + span * rng.rand(N, self.dim)
                F = np.array([problem(x) for x in X])
                used += N
                stagnation = 0
                continue

            # build offspring
            parents = self._select_parents(X, F, ranks, 2 * N)
            children = []
            for i in range(0, 2 * N, 2):
                c = self._sbx_crossover(X[parents[i]], X[parents[i + 1]], lo, hi)
                # mutation
                if rng.rand() < 1.0 / self.dim:
                    d = rng.randint(self.dim)
                    c[d] = lo[d] + span[d] * rng.rand()
                c = np.clip(c, lo, hi)
                children.append(c)
            children = np.array(children)
            Cf = np.array([problem(x) for x in children])
            used += N

            # combine, select N
            XX = np.vstack([X, children])
            FF = np.vstack([F, Cf])
            fronts, ranks = self._fast_non_dominated(FF)
            new_idx = []
            for fr in fronts:
                if len(new_idx) + len(fr) <= N:
                    new_idx.extend(fr)
                else:
                    need = N - len(new_idx)
                    cd = self._crowding(FF, fr)
                    order = np.argsort(-cd)
                    new_idx.extend(fr[order[:need]])
                    break
            X = XX[new_idx]
            F = FF[new_idx]

            # track best (dominated set)
            f0, _ = self._fast_non_dominated(F)
            best = F[f0]

        return X
# EVOLVE-BLOCK-END
