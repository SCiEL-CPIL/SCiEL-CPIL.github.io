# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondom_rank(self, F):
        n = len(F)
        rank = np.full(n, -1, dtype=int)
        S = [set() for _ in range(n)]
        ncount = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    ncount[j] += 1
        current = [i for i in range(n) if ncount[i] == 0]
        r = 0
        while current:
            for i in current:
                rank[i] = r
            nxt = []
            for i in current:
                for j in S[i]:
                    ncount[j] -= 1
                    if ncount[j] == 0:
                        nxt.append(j)
            current = nxt
            r += 1
        return rank

    def _crowding(self, F, idx):
        m = len(idx)
        cd = np.zeros(m, dtype=float)
        if m <= 2:
            return cd
        for k in range(self.n_obj):
            vals = F[idx, k]
            lo = vals.min()
            hi = vals.max()
            rng = hi - lo
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if rng > 1e-30:
                for t in range(1, m - 1):
                    cd[order[t]] += (vals[order[t + 1]] - vals[order[t - 1]]) / rng
        return cd

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 275  # safety margin under 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        pop = int(min(24, max(8, budget // 8)))

        X = lo + span * np.random.rand(pop, dim)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop

        sigma0 = 0.25
        sigma = sigma0
        sigma_decay = 0.998

        while used < budget:
            if time.time() > deadline:
                break

            rank = self._nondom_rank(F)
            # crowding within each rank
            cd = np.zeros(len(F), dtype=float)
            for r in np.unique(rank):
                idx = np.where(rank == r)[0]
                cd[idx] = self._crowding(F, idx)

            # tournament selection: lower rank better, else higher crowding
            def select():
                a = int(np.random.randint(len(F)))
                b = int(np.random.randint(len(F)))
                if rank[a] != rank[b]:
                    return a if rank[a] < rank[b] else b
                return a if cd[a] > cd[b] else b

            p1 = select()
            p2 = select()

            # adaptive mutation
            sigma *= sigma_decay
            child = X[p1] + sigma * span * np.random.randn(dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # replace the worst individual by (rank, -crowding)
            score = rank.astype(float) - cd * 1e-6
            worst = int(np.argmax(score))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
