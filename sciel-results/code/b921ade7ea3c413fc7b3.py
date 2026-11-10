"""Multi-objective optimizer on RE34: NSGA-II style with time guard + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II style multi-objective optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0  # safety margin before the 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def time_ok():
            return time.time() < deadline

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [0] * n
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    rank[p] = 0
                    fronts.append([p])
            i = 0
            while i < len(fronts):
                current = fronts[i]
                nxt = []
                for p in current:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            rank[q] = i + 1
                            nxt.append(q)
                i += 1
                if nxt:
                    fronts.append(nxt)
            return rank

        def crowding_distance(F, idx):
            n = len(idx)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            Fsub = F[idx]
            for m in range(n_obj):
                vals = Fsub[:, m]
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vmax - vmin)
            return cd

        # --- initial population ---
        rng = np.random.default_rng(0)
        pop = int(min(40, max(20, budget // 4)))
        X = lo + span * rng.random((pop, dim))
        F = np.array([np.asarray(problem(x), dtype=float) for x in X], dtype=float)
        used = pop

        while used < budget and time_ok():
            # environmental selection: build the new population of size `pop`
            rank = fast_non_dominated_sort(F)
            fronts = [[] for _ in range(max(rank) + 1)]
            for i in range(pop):
                fronts[rank[i]].append(i)

            new_idx = []
            for fr in fronts:
                if len(new_idx) + len(fr) <= pop:
                    new_idx.extend(fr)
                else:
                    need = pop - len(new_idx)
                    cd = crowding_distance(F, fr)
                    order = np.argsort(-cd)
                    new_idx.extend([fr[j] for j in order[:need]])
                    break

            X = X[new_idx]
            F = F[new_idx]

            # generate offspring
            child = np.empty_like(X)
            childF = np.empty_like(F)
            for j in range(pop):
                # binary tournament on rank (lower better), then crowding
                a = rng.integers(pop)
                b = rng.integers(pop)
                if rank[a] < rank[b]:
                    pa = X[a]
                elif rank[b] < rank[a]:
                    pa = X[b]
                else:
                    cd = crowding_distance(F, list(range(pop)))
                    pa = X[a] if cd[a] >= cd[b] else X[b]
                c = rng.integers(pop)
                d = rng.integers(pop)
                if rank[c] < rank[d]:
                    pc = X[c]
                elif rank[d] < rank[c]:
                    pc = X[d]
                else:
                    cd = crowding_distance(F, list(range(pop)))
                    pc = X[c] if cd[c] >= cd[d] else X[d]

                # SBX crossover
                u = rng.random()
                if u <= 0.9:
                    eta = 15.0
                    beta = np.where(rng.random() <= 0.5,
                                    (2.0 * rng.random()) ** (1.0 / (eta + 1.0)),
                                    (1.0 / (2.0 * rng.random())) ** (1.0 / (eta + 1.0)))
                    child[j] = 0.5 * ((1 + beta) * pa + (1 - beta) * pc)
                else:
                    child[j] = pa.copy()
                # polynomial mutation
                if rng.random() < 0.9:
                    eta_m = 20.0
                    mask = rng.random(dim) < 0.2
                    if np.any(mask):
                        delta = np.where(rng.random(dim) <= 0.5,
                                         (2.0 * rng.random(dim)) ** (1.0 / (eta_m + 1.0)) - 1.0,
                                         1.0 - (2.0 * rng.random(dim)) ** (1.0 / (eta_m + 1.0)))
                        child[j][mask] += delta[mask] * span[mask]
                child[j] = np.clip(child[j], lo, hi)
                childF[j] = np.asarray(problem(child[j]), dtype=float)
                used += 1
                if not time_ok():
                    break

            # combine and keep only the portion actually evaluated
            if used >= budget or not time_ok():
                # stop generating; merge what we have
                n_off = min(pop, int(used - pop))
                if n_off > 0:
                    X = np.vstack([X, child[:n_off]])
                    F = np.vstack([F, childF[:n_off]])
                    rank = fast_non_dominated_sort(F)
                    fronts = [[] for _ in range(max(rank) + 1)]
                    for i in range(len(F)):
                        fronts[rank[i]].append(i)
                    new_idx = []
                    for fr in fronts:
                        if len(new_idx) + len(fr) <= pop:
                            new_idx.extend(fr)
                        else:
                            need = pop - len(new_idx)
                            cd = crowding_distance(F, fr)
                            order = np.argsort(-cd)
                            new_idx.extend([fr[j] for j in order[:need]])
                            break
                    X = X[new_idx]
                    F = F[new_idx]
                break
            X = np.vstack([X, child])
            F = np.vstack([F, childF])

        # --- final local polish of non-dominated points ---
        if time_ok():
            rank = fast_non_dominated_sort(F)
            nd_idx = [i for i in range(len(F)) if rank[i] == 0]
            # pick a handful of the most spread non-dominated points
            if len(nd_idx) > 8:
                cd = crowding_distance(F, nd_idx)
                order = np.argsort(-cd)
                nd_idx = [nd_idx[j] for j in order[:8]]
            for i in nd_idx:
                if not time_ok():
                    break
                x0 = X[i]
                try:
                    res = minimize(lambda v: float(np.sum(np.asarray(problem(v), dtype=float))),
                                   x0, method='Nelder-Mead',
                                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 40})
                    used += int(res.nfev)
                    if used > budget:
                        break
                    xnew = np.clip(res.x, lo, hi)
                    fnew = np.asarray(problem(xnew), dtype=float)
                    # only replace if it is not worse (dominates or equal)
                    if np.all(fnew <= F[i]) and np.any(fnew < F[i]):
                        X[i] = xnew
                        F[i] = fnew
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
