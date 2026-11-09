# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective differential-evolution search with NSGA-II-style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- vectorized helpers ------------------------------------------------
    @staticmethod
    def _dominates(A, B):
        """A: (m,k), B: (n,k) -> (m,n) True if A[i] dominates B[j]."""
        le = (A[:, None, :] <= B[None, :, :])
        lt = (A[:, None, :] < B[None, :, :])
        return (le.all(axis=2)) & (lt.any(axis=2))

    @staticmethod
    def _fast_nondominated_sort(F):
        """Return (fronts, rank) where fronts is a list of index arrays, rank is (N,)."""
        N = len(F)
        S = [[] for _ in range(N)]
        nd = np.zeros(N, dtype=int)
        rank = np.full(N, -1, dtype=int)
        fronts = []
        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                if Optimizer._dominates(F[p], F[q]):
                    S[p].append(q)
                    nd[q] += 1
                elif Optimizer._dominates(F[q], F[p]):
                    nd[p] += 1
        zero = [i for i in range(N) if nd[i] == 0]
        for i in zero:
            rank[i] = 0
        fronts.append(np.array(zero, dtype=int))
        I = 0
        while zero:
            nxt = []
            for p in zero:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        rank[q] = I + 1
                        nxt.append(q)
            I += 1
            zero = nxt
            if zero:
                fronts.append(np.array(zero, dtype=int))
        return fronts, rank

    @staticmethod
    def _crowding(F, idx):
        """Crowding distances for points in F[idx]; returns (len(idx),)."""
        idx = list(idx)
        n = len(idx)
        if n <= 2:
            return np.full(n, np.inf)
        sub = F[idx]
        cd = np.zeros(n)
        for m in range(sub.shape[1]):
            vals = sub[:, m]
            srt = np.argsort(vals)
            lo, hi = vals[srt[0]], vals[srt[-1]]
            if hi <= lo:
                continue
            cd[srt[0]] = np.inf
            cd[srt[-1]] = np.inf
            for i in range(1, n - 1):
                cd[srt[i]] += (vals[srt[i + 1]] - vals[srt[i - 1]]) / (hi - lo)
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        N = max(12, min(40, self.budget // 5))
        # initial population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        FCR = 0.8
        CR = 0.9
        Fm = 0.5

        while used < self.budget:
            if time.time() > deadline:
                break
            # fast non-dominated sort
            fronts, rank = self._fast_nondominated_sort(F)
            # pick parents from front 0 (crowding)
            f0 = fronts[0]
            cd0 = self._crowding(F, f0)
            order = f0[np.argsort(-cd0)]
            p1 = order[rng.integers(len(order))]
            p2 = order[rng.integers(len(order))]
            p3 = order[rng.integers(len(order))]
            v = X[p1] + FCR * (X[p2] - X[p3])
            # binomial crossover
            r = rng.integers(self.dim)
            cross = rng.random(self.dim) < CR
            cross[r] = True
            u = np.where(cross, v, X[p1])
            u = np.clip(u, lo, hi)
            fu = problem(u)
            used += 1
            # tournament selection for replacement: pick worst by (rank, -crowd)
            # assign crowding per front for a simple fitness
            crowd = np.zeros(N)
            for fr in fronts:
                c = self._crowding(F, fr)
                crowd[fr] = c
            # fitness: lower rank better, then higher crowd better
            score = rank.astype(float) - crowd
            worst = int(np.argmax(score))
            X[worst] = u
            F[worst] = fu
        return X
# EVOLVE-BLOCK-END
