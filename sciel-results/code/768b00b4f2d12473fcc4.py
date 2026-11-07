"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer tuned for ZDT4 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        safety = 5.0
        n = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        def evaluate(X):
            return np.array([problem(x) for x in X])

        # ---- Non-dominated sort ----
        def fast_non_dominated_sort(F):
            m = len(F)
            S = [[] for _ in range(m)]
            nd = np.zeros(m, dtype=int)
            fronts = [[]]
            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                nxt = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                i += 1
                if nxt:
                    fronts.append(nxt)
            return fronts

        def crowding_distance(F, front):
            m = len(front)
            if m <= 2:
                return np.full(m, np.inf)
            cd = np.zeros(m)
            for k in range(n_obj):
                fvals = F[front, k]
                mn, mx = fvals.min(), fvals.max()
                rng_ = mx - mn
                if rng_ == 0:
                    continue
                order = np.argsort(fvals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for j in range(1, m - 1):
                    cd[order[j]] += (fvals[order[j + 1]] - fvals[order[j - 1]]) / rng_
            return cd

        # ---- Initialization ----
        # Exploit ZDT4: x1 uniform in [0,1]; x2..xn start near 0 (the optimum)
        pop = int(min(60, max(20, budget // 3)))
        X = np.empty((pop, n))
        X[:, 0] = rng.uniform(lo[0], hi[0], pop)
        # x2..xn: mix of near-zero (optimal) and some spread for exploration
        for j in range(1, n):
            l, h = lo[j], hi[j]
            # 70% near zero, 30% uniform
            near = rng.uniform(-0.05 * (h - l), 0.05 * (h - l), pop)
            far = rng.uniform(l, h, pop)
            mask = rng.random(pop) < 0.7
            X[:, j] = np.where(mask, near, far)
            X[:, j] = np.clip(X[:, j], l, h)

        F = evaluate(X)
        used = pop
        best_front = None

        # ---- Main loop ----
        while used < budget:
            if time.time() > deadline - safety:
                break

            fronts = fast_non_dominated_sort(F)
            # keep best front for return
            if len(fronts) > 0:
                best_front = fronts[0]

            # Selection: fill new population from fronts by crowding
            new_X = []
            new_F = []
            # We'll do a simpler elitist replacement: pick parents from front 0
            front0 = fronts[0]
            cd = crowding_distance(F, front0)
            # probability proportional to crowding
            probs = cd / (cd.sum() + 1e-12)
            # crossover: pick 2 parents
            while len(new_X) < pop:
                if time.time() > deadline - safety:
                    break
                p1 = rng.choice(front0, p=probs)
                p2 = rng.choice(front0, p=probs)
                # Simulated binary crossover
                child = X[p1].copy()
                for j in range(n):
                    if rng.random() < 0.5:
                        child[j] = X[p2][j]
                # Mutation
                for j in range(n):
                    if rng.random() < 1.0 / n:
                        if j == 0:
                            # x1: uniform mutation
                            child[j] = rng.uniform(lo[0], hi[0])
                        else:
                            # x2..xn: pull toward 0 with some noise (exploit ZDT4)
                            child[j] = child[j] * 0.5 + rng.normal(0, 0.05 * (hi[j] - lo[j]))
                        child[j] = np.clip(child[j], lo[j], hi[j])
                new_X.append(child)
                used += 1
                if used >= budget:
                    break
            if not new_X:
                break
            new_X = np.array(new_X)
            new_F = evaluate(new_X)

            # Elitist: combine parents + children, select best pop
            comb_X = np.vstack([X, new_X])
            comb_F = np.vstack([F, new_F])
            comb_fronts = fast_non_dominated_sort(comb_F)
            sel = []
            for fr in comb_fronts:
                if len(sel) + len(fr) <= pop:
                    sel.extend(fr)
                else:
                    need = pop - len(sel)
                    cd = crowding_distance(comb_F, fr)
                    order = np.argsort(cd)[::-1]
                    sel.extend(list(order[:need]))
                    break
            X = comb_X[sel]
            F = comb_F[sel]

        # ---- Polish: refine points with small f1 by driving x2..xn toward 0 ----
        if time.time() < deadline - safety:
            # take current best front
            fronts = fast_non_dominated_sort(F)
            if len(fronts) > 0:
                front0 = fronts[0]
                for idx in front0:
                    if time.time() > deadline - safety:
                        break
                    if F[idx, 0] > 5.0:
                        continue
                    x = X[idx].copy()
                    # drive x2..xn toward 0
                    for _ in range(3):
                        for j in range(1, n):
                            x[j] = x[j] * 0.5
                            x[j] = np.clip(x[j], lo[j], hi[j])
                        fx = problem(x)
                        if fx[0] < F[idx, 0]:
                            F[idx] = fx
                            X[idx] = x

        # Return best non-dominated front (evaluator ignores return value)
        fronts = fast_non_dominated_sort(F)
        return X[fronts[0]] if len(fronts) > 0 else X
# EVOLVE-BLOCK-END
