"""Multi-objective optimizer on WFG4 (3 objectives, 12 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style evolutionary multi-objective search with a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        DEADLINE = start + 240.0  # safety margin well under the 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def eval_batch(X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            vals = [problem(x) for x in X]
            return np.array(vals)

        # ---- population sizing ----
        pop = int(min(50, max(20, budget // 4)))

        # ---- fast non-dominated sorting ----
        def fast_nondominated_sort(F):
            n = F.shape[0]
            S = [set() for _ in range(n)]
            ndom = np.zeros(n, dtype=int)
            fronts = []
            front = set()
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        ndom[p] += 1
                if ndom[p] == 0:
                    front.add(p)
                    fronts.append(front)
            i = 0
            while fronts[i]:
                next_front = set()
                for p in fronts[i]:
                    for q in S[p]:
                        ndom[q] -= 1
                        if ndom[q] == 0:
                            next_front.add(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        # ---- crowding distance ----
        def crowding_distance(F):
            n = F.shape[0]
            cd = np.zeros(n)
            for m in range(n_obj):
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if n > 2 and F[order[-1], m] > F[order[0], m]:
                    for k in range(1, n - 1):
                        cd[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / (
                            F[order[-1], m] - F[order[0], m])
            return cd

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # ---- SBX crossover ----
        def sbx_crossover(a, b, eta=20.0):
            u = rng.random(dim)
            child1 = np.empty(dim)
            child2 = np.empty(dim)
            for j in range(dim):
                if u[j] <= 0.5:
                    beta = (2.0 * u[j]) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u[j]))) ** (1.0 / (eta + 1.0))
                child1[j] = 0.5 * ((1.0 + beta) * a[j] + (1.0 - beta) * b[j])
                child2[j] = 0.5 * ((1.0 - beta) * a[j] + (1.0 + beta) * b[j])
            return child1, child2

        # ---- polynomial mutation ----
        def mutation(x, sigma, eta=20.0):
            mask = rng.random(dim) < 0.3
            if not np.any(mask):
                return x
            y = x.copy()
            for j in np.where(mask)[0]:
                u = rng.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                y[j] += delta * sigma
            return y

        # ---- initialization ----
        X = lo + (hi - lo) * rng.random((pop, dim))
        F = eval_batch(X)
        used = pop

        # ---- main loop ----
        gen = 0
        while used < budget:
            if time.time() > DEADLINE:
                break
            fronts = fast_nondominated_sort(F)
            # build selection pool (fronts in order, tie-break by crowding)
            pool = []
            for fr in fronts:
                idx = list(fr)
                cd = crowding_distance(F[idx])
                order = np.argsort(-cd)
                pool.extend(idx[order])
            pool = np.array(pool[:pop])

            # binary tournament on the ranked pool
            rank_of = {int(p): r for r, p in enumerate(pool)}

            def tournament():
                i1, i2 = rng.integers(0, len(pool), 2)
                return int(pool[i1]) if rank_of[int(pool[i1])] <= rank_of[int(pool[i2])] else int(pool[i2])

            new_X = []
            new_F = []
            while len(new_X) < pop:
                if used + 1 > budget:
                    break
                p1 = tournament()
                p2 = tournament()
                if p1 == p2:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()
                else:
                    c1, c2 = sbx_crossover(X[p1], X[p2])
                # adaptive sigma: larger early, smaller later
                sigma = (hi - lo) * (0.15 * (1.0 - used / max(1, budget)) + 0.02)
                c1 = np.clip(mutation(c1, sigma), lo, hi)
                c2 = np.clip(mutation(c2, sigma), lo, hi)
                new_X.append(c1)
                new_X.append(c2)
                used += 2
            new_X = np.array(new_X[:pop])
            new_F = eval_batch(new_X)

            # combine and select
            all_X = np.vstack([X, new_X])
            all_F = np.vstack([F, new_F])
            fronts = fast_nondominated_sort(all_F)
            selected = []
            for fr in fronts:
                if len(selected) + len(fr) <= pop:
                    selected.extend(list(fr))
                else:
                    idx = list(fr)
                    cd = crowding_distance(all_F[idx])
                    order = np.argsort(-cd)
                    need = pop - len(selected)
                    selected.extend(idx[order[:need]])
                    break
            X = all_X[np.array(selected)]
            F = all_F[np.array(selected)]
            gen += 1

        return X
# EVOLVE-BLOCK-END
