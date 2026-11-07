"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominance(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1
        return S, n_dom

    def _nondominated_sort(self, F):
        n = len(F)
        S, n_dom = self._dominance(F)
        fronts = []
        current = [i for i in range(n) if n_dom[i] == 0]
        while current:
            fronts.append(current[:])
            next_front = []
            for i in current:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current = next_front
        return fronts

    def _crowding(self, F, idxs):
        n = len(idxs)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        for k in range(self.n_obj):
            vals = F[idxs, k]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            rng_ = vals[order[-1]] - vals[order[0]]
            if rng_ < 1e-12:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (vals[order[i+1]] - vals[order[i-1]]) / rng_
        return cd

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        while used < self.budget:
            fronts = self._nondominated_sort(F)
            
            # Assign fitness: (front_index, -crowding)
            fitness = [None] * N
            for fi, front in enumerate(fronts):
                cd = self._crowding(F, front)
                for ci, idx in enumerate(front):
                    fitness[idx] = (fi, -cd[ci])
            
            # Tournament selection
            def select():
                a, b = rng.integers(0, N, size=2)
                return a if fitness[a] < fitness[b] else b
            
            # Create offspring
            off_X = []
            off_F = []
            n_off = min(N, self.budget - used)
            for _ in range(n_off):
                p1, p2 = select(), select()
                # BLX-alpha style crossover
                child = X[p1].copy()
                for i in range(self.dim):
                    if rng.rand() < 0.5:
                        child[i] = X[p2][i]
                # Mutation
                if rng.rand() < 0.8:
                    m_idx = rng.integers(self.dim)
                    child[m_idx] = lo[m_idx] + (hi[m_idx] - lo[m_idx]) * rng.random()
                child = np.clip(child, lo, hi)
                off_X.append(child)
                off_F.append(problem(child))
                used += 1
            
            if not off_X:
                break
            
            off_F = np.array(off_F)
            X_all = np.vstack([X, np.array(off_X)])
            F_all = np.vstack([F, off_F])
            
            # Select next gen
            fronts_all = self._nondominated_sort(F_all)
            next_X, next_F = [], []
            for front in fronts_all:
                if len(next_X) + len(front) <= N:
                    for idx in front:
                        next_X.append(X_all[idx])
                        next_F.append(F_all[idx])
                else:
                    cd = self._crowding(F_all, front)
                    order = np.argsort(cd)[::-1]
                    for idx in order:
                        if len(next_X) < N:
                            next_X.append(X_all[idx])
                            next_F.append(F_all[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X
# EVOLVE-BLOCK-END
