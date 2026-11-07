import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        pop_size = min(30, self.budget // 3)
        X = lo + rng_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def nds(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_d = [0] * n
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                        n_d[q] += 1
            current = [p for p in range(n) if n_d[p] == 0]
            if current:
                fronts.append(current)
            while current:
                nxt = []
                for p in current:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            nxt.append(q)
                if nxt:
                    fronts.append(nxt)
                current = nxt
            return fronts

        def cd(F, idxs):
            n = len(idxs)
            if n <= 2:
                return [float('inf')] * n
            d = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[i][m] for i in idxs]
                mn, mx = min(vals), max(vals)
                if mx == mn:
                    continue
                srt = sorted(idxs, key=lambda i: F[i][m])
                pm = {idx: pos for pos, idx in enumerate(srt)}
                d[pm[srt[0]]] = float('inf')
                d[pm[srt[-1]]] = float('inf')
                for k in range(1, n - 1):
                    i = srt[k]
                    d[pm[i]] += (F[srt[k + 1]][m] - F[srt[k - 1]][m]) / (mx - mn)
            return d

        # Main loop
        mut_scale = 0.3
        while used < self.budget:
            if time.time() > deadline:
                break

            fronts = nds(F)
            if not fronts:
                break
            ff = fronts[0]
            dists = cd(F, ff)

            # Build selection pool: first front, plus second if small
            if len(ff) < 6 and len(fronts) > 1:
                sf = fronts[1]
                dists2 = cd(F, sf)
                pool = ff + sf
                pd = list(dists) + [x * 0.5 for x in dists2]
            else:
                pool = ff
                pd = dists

            mx_pd = max(pd)
            if mx_pd == 0:
                probs = np.ones(len(pool)) / len(pool)
            else:
                probs = np.array(pd) / sum(pd)

            i1 = np.random.choice(len(pool), p=probs)
            i2 = np.random.choice(len(pool), p=probs)
            p1, p2 = X[pool[i1]], X[pool[i2]]

            # SBX-like crossover
            a = np.random.uniform(0.2, 0.8)
            child = a * p1 + (1 - a) * p2

            # Polynomial mutation
            mut = np.random.standard_normal(self.dim) * (mut_scale * rng_)
            child = np.clip(child + mut, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement
            dom_idx = [i for i in range(len(F)) if np.all(fc <= F[i]) and np.any(fc < F[i])]
            if dom_idx:
                worst = max(dom_idx, key=lambda i: np.sum(F[i]))
                X[worst] = child
                F[worst] = fc
            else:
                is_dom = any(np.all(F[i] <= fc) and np.any(F[i] < fc) for i in range(len(F)))
                if is_dom:
                    # Replace most crowded in first front
                    if ff:
                        mi = ff[int(np.argmax(dists))]
                        X[mi] = child
                        F[mi] = fc
                else:
                    if ff:
                        mi = ff[int(np.argmax(dists))]
                        X[mi] = child
                        F[mi] = fc

        return X
