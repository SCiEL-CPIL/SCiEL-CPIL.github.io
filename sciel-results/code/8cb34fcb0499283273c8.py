"""Multi-objective optimizer on RE37: NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin before the 300 s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        pop = int(min(20, max(8, self.budget // 4)))
        # Initial population (Sobol-like via Halton for better coverage)
        X = self._halton_init(pop, self.dim, lo, hi, rng)
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            nd = [0] * n
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts.append([])
                    fronts[-1].append(p)
            i = 0
            while i < len(fronts):
                current = fronts[i]
                nxt = []
                for p in current:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                i += 1
                if nxt:
                    fronts.append(nxt)
            return fronts

        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return [float('inf')] * m
            cd = np.zeros(m)
            Fsub = F[indices]
            for k in range(self.n_obj):
                order = np.argsort(Fsub[:, k])
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                rng_k = Fsub[order[-1], k] - Fsub[order[0], k]
                if rng_k == 0:
                    continue
                for j in range(1, m - 1):
                    cd[order[j]] += (Fsub[order[j + 1], k] - Fsub[order[j - 1], k]) / rng_k
            return cd

        def tournament_select(fronts, pop_size, rng):
            # assign ranks
            rank = np.zeros(len(F), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi + 1
            parents = []
            for _ in range(pop_size):
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if rank[a] < rank[b]:
                    parents.append(a)
                elif rank[b] < rank[a]:
                    parents.append(b)
                else:
                    parents.append(a if rng.random() < 0.5 else b)
            return parents

        while used < self.budget:
            if time.perf_counter() > deadline:
                break
            fronts = fast_non_dominated_sort(F)
            # crowding on the first front for diversity
            first_front = fronts[0]
            cd_first = crowding_distance(F, first_front)
            cd_map = {}
            for j, idx in enumerate(first_front):
                cd_map[idx] = cd_first[j]

            def better(a, b):
                ra, rb = rank[a], rank[b]
                if ra != rb:
                    return ra < rb
                if a in cd_map and b in cd_map:
                    return cd_map[a] > cd_map[b]
                return a < b

            # generate offspring
            parents = tournament_select(fronts, pop, rng)
            children = []
            for i in range(pop):
                p1 = parents[i]
                p2 = parents[(i + 1) % pop]
                # SBX-like crossover with polynomial blending
                u = rng.random()
                if u < 0.9:
                    x1, x2 = X[p1], X[p2]
                    # blend
                    c1 = 0.5 * (x1 + x2) + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    c2 = 0.5 * (x1 + x2) - 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    children.append(np.clip(c1, lo, hi))
                    children.append(np.clip(c2, lo, hi))
                else:
                    children.append(X[p1].copy())
                    children.append(X[p2].copy())
            # dedupe to pop size
            if len(children) > pop:
                keep = rng.choice(len(children), size=pop, replace=False)
                children = [children[k] for k in keep]
            else:
                while len(children) < pop:
                    children.append(X[rng.integers(len(X))].copy())

            Fc = np.array([np.asarray(problem(c), dtype=float) for c in children])
            used += len(children)

            # combine parents + children
            Xall = np.vstack([X, np.array(children)])
            Fall = np.vstack([F, Fc])
            fronts_all = fast_non_dominated_sort(Fall)

            new_X, new_F = [], []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop:
                    for idx in front:
                        new_X.append(Xall[idx])
                        new_F.append(Fall[idx])
                else:
                    # need crowding to select the rest
                    rem = pop - len(new_X)
                    cd = crowding_distance(Fall, front)
                    order = np.argsort(-cd)
                    for j in range(rem):
                        idx = front[order[j]]
                        new_X.append(Xall[idx])
                        new_F.append(Fall[idx])
                    break
            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _halton_init(self, n, dim, lo, hi, rng):
        X = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                base = d + 2  # primes 2,3,5,...
                f = 1.0
                t = float(i + 1)
                while t > 0:
                    f /= base
                    t = np.floor(t)
                    t -= np.floor(t) * 0  # keep as float
                    t = int(t)
                # simpler: compute halton properly
                val = 0.0
                f = 1.0
                k = i + 1
                while k > 0:
                    f /= base
                    val += f * (k % base)
                    k //= base
                X[i, d] = lo[d] + (hi[d] - lo[d]) * val
        return X
# EVOLVE-BLOCK-END
