"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary search with a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    # ---- helper: fast non-dominated sorting (returns rank array) ----
    @staticmethod
    def _ranks(F):
        n = F.shape[0]
        rank = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]          # S[i] = indices dominated by i
        ndom = np.zeros(n, dtype=int)       # number of individuals dominating i
        front = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    ndom[i] += 1
            if ndom[i] == 0:
                rank[i] = 0
                front.append(i)
        k = 0
        while front:
            nxt = []
            for i in front:
                for j in S[i]:
                    ndom[j] -= 1
                    if ndom[j] == 0:
                        rank[j] = k + 1
                        nxt.append(j)
            front = nxt
            k += 1
        return rank

    # ---- helper: crowding distance ----
    @staticmethod
    def _crowding(F, idx):
        m = len(idx)
        if m <= 2:
            return np.full(m, np.inf)
        cd = np.zeros(m)
        Fsub = F[idx]
        nobj = Fsub.shape[1]
        for o in range(nobj):
            order = np.argsort(Fsub[:, o])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            span = Fsub[order[-1], o] - Fsub[order[0], o]
            if span > 1e-12:
                for t in range(1, m - 1):
                    cd[order[t]] += (Fsub[order[t + 1], o] - Fsub[order[t - 1], o]) / span
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0  # safety margin under the 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # population size: enough to cover the 3D front, bounded by budget
        pop = int(min(40, max(12, self.budget // 5)))

        # ---- initial population: Latin-hypercube-style scatter for diversity ----
        X = np.empty((pop, self.dim))
        for d in range(self.dim):
            X[:, d] = lo[d] + span[d] * (np.arange(pop) + rng.random(pop)) / pop
        F = np.array([problem(x) for x in X])
        used = pop

        eta_c = 15.0   # SBX distribution index
        eta_m = 20.0   # mutation distribution index
        p_mut = 1.0 / self.dim

        while used + pop <= self.budget and (time.time() - t0) < DEADLINE:
            # ranks + crowding for selection
            rank = self._ranks(F)
            # build selection pool ordered by (rank, -crowding) within rank
            # (tournament on rank then crowding)
            nfronts = rank.max() + 1
            cd = np.zeros(pop)
            for r in range(nfronts):
                idx = np.where(rank == r)[0]
                cd[idx] = self._crowding(F, idx)
            # combine into a single selection key: rank primary, -crowding secondary
            # normalize crowding to [0,1] within whole pop for a scalar key
            cdn = cd
            if cdn.max() > cdn.min():
                cdn = (cdn - cdn.min()) / (cdn.max() - cdn.min())
            key = rank - cdn  # lower is better

            # ---- generate offspring via SBX crossover + polynomial mutation ----
            Xc = np.empty((pop, self.dim))
            Fc = np.empty((pop, self.n_obj))
            for i in range(0, pop, 2):
                # binary tournament on key
                a, b = rng.integers(0, pop, size=2)
                p1 = X[a] if key[a] <= key[b] else X[b]
                a, b = rng.integers(0, pop, size=2)
                p2 = X[a] if key[a] <= key[b] else X[b]

                u = rng.random(self.dim)
                alpha = 2.0 * np.minimum(u, 1.0 - u)
                beta1 = (alpha + 1.0) ** (-1.0 / (eta_c + 1.0))
                alpha = 2.0 * np.minimum(1.0 - u, u)
                beta2 = (alpha + 1.0) ** (-1.0 / (eta_c + 1.0))

                c1 = 0.5 * ((1.0 + 1.0 / beta1) * p1 + (1.0 - 1.0 / beta1) * p2)
                c2 = 0.5 * ((1.0 + 1.0 / beta2) * p2 + (1.0 - 1.0 / beta2) * p1)

                # polynomial mutation
                mask = rng.random(self.dim) < p_mut
                if np.any(mask):
                    u = rng.random(self.dim)
                    delta1 = (1.0 + 2.0 * (u - 0.0)) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (1.0 + 2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    c1 = c1 + mask * (c1 - lo) * delta1
                    c2 = c2 + mask * (hi - c2) * delta2

                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                Xc[i] = c1
                Xc[i + 1] = c2
                Fc[i] = problem(c1)
                Fc[i + 1] = problem(c2)
                used += 2

            # ---- environmental selection: combine parents + offspring ----
            Xall = np.vstack([X, Xc])
            Fall = np.vstack([F, Fc])
            rank = self._ranks(Fall)
            nfronts = rank.max() + 1
            cd = np.zeros(len(Xall))
            for r in range(nfronts):
                idx = np.where(rank == r)[0]
                cd[idx] = self._crowding(Fall, idx)
            if cd.max() > cd.min():
                cdn = (cd - cd.min()) / (cd.max() - cd.min())
            else:
                cdn = cd
            key = rank - cdn
            order = np.argsort(key, kind="stable")
            X = Xall[order[:pop]]
            F = Fall[order[:pop]]

        # ---- final deadline guard: return best non-dominated set ----
        return X
# EVOLVE-BLOCK-END
