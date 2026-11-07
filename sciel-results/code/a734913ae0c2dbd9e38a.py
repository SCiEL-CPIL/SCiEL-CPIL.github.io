"""NSGA-II + Differential-Evolution multi-objective optimizer for RE34 (5D, 3 obj)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II with DE mutation, SBX crossover, polynomial mutation, deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _fast_non_dominated_sort(F):
        n = F.shape[0]
        S = [[] for _ in range(n)]
        nd = [0] * n
        front = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if F[p].tolist() <= F[q].tolist() and F[p].tolist() < F[q].tolist():
                    S[p].append(q)
                elif F[q].tolist() <= F[p].tolist() and F[q].tolist() < F[p].tolist():
                    nd[p] += 1
            if nd[p] == 0:
                front.append(p)
        ranks = [-1] * n
        i = 0
        while front:
            for p in front:
                ranks[p] = i
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            front = next_front
            i += 1
        return ranks

    @staticmethod
    def _crowding_distance(F, idx):
        m = len(idx)
        cd = [0.0] * m
        if m <= 2:
            return [float('inf')] * m
        for k in range(F.shape[1]):
            vals = np.array([F[i, k] for i in idx])
            order = np.argsort(vals)
            cd[order[0]] = float('inf')
            cd[order[-1]] = float('inf')
            rng = vals[order[-1]] - vals[order[0]]
            if rng <= 0:
                continue
            for j in range(1, m - 1):
                cd[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / rng
        return cd

    @staticmethod
    def _dominates(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _sbx(self, x1, x2, lo, hi, eta):
        if np.random.rand() <= 0.5:
            c1, c2 = x1.copy(), x2.copy()
        else:
            c1, c2 = x2.copy(), x1.copy()
        for i in range(self.dim):
            if abs(x1[i] - x2[i]) > 1e-14:
                x1i, x2i = x1[i], x2[i]
                if x2i > x1i:
                    x1i, x2i = x2i, x1i
                if np.random.rand() <= 0.5:
                    beta = 1.0 + 2.0 * (x1i - lo[i]) / (x2i - x1i)
                    alpha = 2.0 - beta ** (eta + 1.0)
                    rand = np.random.rand()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    c1[i] = 0.5 * ((x1i + x2i) - betaq * (x2i - x1i))
                else:
                    beta = 1.0 + 2.0 * (hi[i] - x2i) / (x2i - x1i)
                    alpha = 2.0 - beta ** (eta + 1.0)
                    rand = np.random.rand()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    c2[i] = 0.5 * ((x1i + x2i) + betaq * (x2i - x1i))
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polymut(self, x, lo, hi, eta):
        x = x.copy()
        for i in range(self.dim):
            if np.random.rand() <= 0.5:
                y = x[i]
                if hi[i] - y > 1e-14 and y - lo[i] > 1e-14:
                    delta1 = (y - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - y) / (hi[i] - lo[i])
                    rand = np.random.rand()
                    if rand <= 0.5:
                        xy = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                        mut = 0.5 * (1.0 - xy ** (1.0 / (eta + 1.0)))
                    else:
                        xy = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                        mut = 0.5 * (xy ** (1.0 / (eta + 1.0)) - 1.0)
                    x[i] = y + mut * (hi[i] - lo[i])
        return np.clip(x, lo, hi)

    def _tournament(self, ranks, cd, idx):
        a = idx[np.random.randint(len(idx))]
        b = idx[np.random.randint(len(idx))]
        if ranks[a] != ranks[b]:
            return a if ranks[a] < ranks[b] else b
        return a if cd[a] >= cd[b] else b

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        np.random.seed(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim, n_obj = self.dim, self.n_obj
        budget = int(self.budget)
        pop = int(min(20, max(2, budget // 10)))
        eta_c, eta_m = 20.0, 20.0
        de_F = 0.7
        deadline = time.time() + 300.0

        X = lo + (hi - lo) * np.random.rand(pop, dim)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        while used + pop <= budget:
            if time.time() > deadline:
                break
            ranks = self._fast_non_dominated_sort(F)
            cd = self._crowding_distance(F, list(range(pop)))

            # DE boost: one DE offspring from a random triple (uses 1 eval)
            i, r1, r2 = (np.random.randint(pop) for _ in range(3))
            while r1 == i:
                r1 = np.random.randint(pop)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pop)
            de_child = X[i] + de_F * (X[r1] - X[r2])
            de_child = np.clip(de_child, lo, hi)
            de_Fc = np.asarray(problem(de_child), dtype=float)
            used += 1

            # build next population (size pop)
            next_X = [de_child]
            next_F = [de_Fc]
            while len(next_X) < pop and used + 1 <= budget:
                p1 = self._tournament(ranks, cd, list(range(pop)))
                p2 = self._tournament(ranks, cd, list(range(pop)))
                c1, c2 = self._sbx(X[p1], X[p2], lo, hi, eta_c)
                c1 = self._polymut(c1, lo, hi, eta_m)
                c2 = self._polymut(c2, lo, hi, eta_m)
                next_X.append(c1); next_F.append(np.asarray(problem(c1), dtype=float)); used += 1
                if len(next_X) < pop and used + 1 <= budget:
                    next_X.append(c2); next_F.append(np.asarray(problem(c2), dtype=float)); used += 1

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
