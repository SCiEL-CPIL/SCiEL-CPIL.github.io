"""Multi-objective optimizer on RE37 (NSGA-II style, fresh design)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj
        budget = int(self.budget)

        def fast_nondom(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            nd = [0] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        nd[j] += 1
            front = [i for i in range(n) if nd[i] == 0]
            r = 0
            while front:
                for i in front:
                    ranks[i] = r
                nxt = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            nxt.append(j)
                front = nxt
                r += 1
            return ranks

        def crowd(F, order):
            n = len(order)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            for k in range(n_obj):
                vals = np.array([F[i, k] for i in order], dtype=float)
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    continue
                idx = np.argsort(vals)
                cd[order[idx[0]]] = np.inf
                cd[order[idx[-1]]] = np.inf
                for t in range(1, n - 1):
                    cd[order[idx[t]]] += (vals[idx[t + 1]] - vals[idx[t - 1]]) / (mx - mn)
            return cd

        # seed-sensitive random initialization (uses global np.random state)
        pop = int(max(10, min(40, budget // 4)))
        X = lo + span * np.random.random((pop, dim))
        F = np.array([np.asarray(problem(x), dtype=float) for x in X], dtype=float)
        used = pop

        while used < budget:
            if time.time() > deadline:
                break
            ranks = fast_nondom(F)
            fronts = {}
            for i in range(len(F)):
                fronts.setdefault(int(ranks[i]), []).append(i)
            for key in fronts:
                cd = crowd(F, fronts[key])
                for pos, i in enumerate(fronts[key]):
                    F[i] = F[i]  # noop
                    # store crowding in a side array via rank trick: attach to X row? use dict
                    fronts[key][pos] = (i, cd[pos])

            # build combined list with (rank, -crowd) for selection
            members = []
            for key in sorted(fronts):
                for i, cdv in fronts[key]:
                    members.append((int(ranks[i]), -cdv, i))
            members.sort()
            order = [m[2] for m in members]

            # tournament selection (size 2)
            def sel():
                a, b = order[np.random.randint(len(order))], order[np.random.randint(len(order))]
                ra, rb = int(ranks[a]), int(ranks[b])
                if ra != rb:
                    return a if ra < rb else b
                # tie-break by crowding (larger better)
                ca = -members[order.index(a)][1]
                cb = -members[order.index(b)][1]
                return a if ca >= cb else b

            # SBX crossover + polynomial mutation
            eta_c, eta_m = 15.0, 20.0
            pc, pm = 0.9, 1.0 / dim

            def cross_mut(x1, x2):
                child = x2.copy()
                if np.random.rand() < pc:
                    child = np.zeros(dim)
                    for d in range(dim):
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                child = np.clip(child, lo, hi)
                if np.random.rand() < pm:
                    for d in range(dim):
                        if np.random.rand() < pm:
                            u = np.random.rand()
                            delta1 = (x2[d] - lo[d]) / span[d] if span[d] > 0 else 0.0
                            delta2 = (hi[d] - x2[d]) / span[d] if span[d] > 0 else 0.0
                            if u < 0.5:
                                xy = 1.0 - delta1
                                mut_pow = 2.0 + (1.0 / xy) * eta_m
                                val = delta1 * ((2 * u) ** (1.0 / (mut_pow + 1)) - 1.0)
                            else:
                                xy = 1.0 - delta2
                                mut_pow = 2.0 + (1.0 / xy) * eta_m
                                val = delta2 * ((2 * (1 - u)) ** (1.0 / (mut_pow + 1)) - 1.0)
                            child[d] = x2[d] + val * span[d]
                            child[d] = np.clip(child[d], lo[d], hi[d])
                return child

            # generate offspring
            n_child = pop
            children_X = []
            for _ in range(n_child):
                p1, p2 = sel(), sel()
                if p1 == p2:
                    p2 = order[np.random.randint(len(order))]
                children_X.append(cross_mut(X[p1], X[p2]))
            children_F = np.array([np.asarray(problem(x), dtype=float) for x in children_X], dtype=float)
            used += n_child

            # combine and select
            PX = np.vstack([X, children_X])
            PF = np.vstack([F, children_F])
            pranks = fast_nondom(PF)
            pfronts = {}
            for i in range(len(PF)):
                pfronts.setdefault(int(pranks[i]), []).append(i)
            pfronts = {k: v for k, v in pfronts.items()}
            for key in pfronts:
                cd = crowd(PF, pfronts[key])
                pfronts[key] = [(i, cd[pos]) for pos, i in enumerate(pfronts[key])]

            new_order = []
            for key in sorted(pfronts):
                lst = sorted(pfronts[key], key=lambda t: -t[1])
                for i, cdv in lst:
                    new_order.append(i)
            new_order = new_order[:pop]
            X = PX[new_order]
            F = PF[new_order]
        return X
# EVOLVE-BLOCK-END
