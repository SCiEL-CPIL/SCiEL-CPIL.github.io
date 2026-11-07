"""Multi-objective optimizer on ZDT3 (NSGA-II with crowding distance + time guard)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective search with a wall-clock safety guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        t0 = time.time()
        DEADLINE = t0 + 320.0   # 10s safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        # ---- fast non-dominated sort (early stop) ----
        def fast_nds(F, K=3):
            n = len(F)
            S = [set() for _ in range(n)]
            nd = [0] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
            fronts = []
            front = [i for i in range(n) if nd[i] == 0]
            if front:
                fronts.append(front)
            idx = 0
            while front and idx < K:
                nxt = []
                for p in front:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                front = nxt
                if front:
                    fronts.append(front)
                idx += 1
            return fronts

        # ---- crowding distance ----
        def crowding(F, idxs):
            if len(idxs) <= 2:
                return np.full(len(idxs), np.inf)
            m = len(idxs)
            cd = np.zeros(m)
            for k in range(n_obj):
                vals = F[idxs, k]
                sv = np.argsort(vals)
                rngv = vals[sv[-1]] - vals[sv[0]]
                if rngv > 1e-14:
                    cd[sv[0]] = np.inf
                    cd[sv[-1]] = np.inf
                    for t in range(1, m - 1):
                        cd[sv[t]] += (vals[sv[t + 1]] - vals[sv[t - 1]]) / rngv
            return cd

        # ---- hypervolume (2-D, for final archive selection) ----
        def hv2(F, ref):
            pts = [f for f in F if np.all(f < ref)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda p: p[0])
            vol = 0.0
            best = ref[1]
            for p in pts:
                if p[1] < best:
                    vol += (ref[0] - p[0]) * (best - p[1])
                    best = p[1]
            return vol

        # ---- ZDT3-aware initial population ----
        def zdt3_front_init():
            X = np.empty((self.dim, dim))
            for i in range(self.dim):
                t = i / (self.dim - 1)
                X[i, 0] = t
                X[i, 1] = 0.1 * t
                X[i, 2] = 0.1 * (1.0 - t)
                if dim > 3:
                    X[i, 3:] = rng.random(dim - 3)
            return X

        # ---- vectorized mutation (single candidate) ----
        def mutate(x):
            m = 0.15
            x = x + m * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            # occasional re-seed of the ZDT3 front variables
            if rng.random() < 0.05:
                t = rng.random()
                x[0] = t
                x[1] = 0.1 * t
                x[2] = 0.1 * (1.0 - t)
                x = np.clip(x, lo, hi)
            return x

        # ---- population management ----
        def manage(X, F, nsel):
            fronts = fast_nds(F, K=3)
            sel = []
            for fr in fronts:
                if len(sel) + len(fr) <= nsel:
                    sel.extend(fr)
                else:
                    need = nsel - len(sel)
                    cd = crowding(F, fr)
                    order = np.argsort(-cd)
                    sel.extend([fr[i] for i in order[:need]])
                    break
            sel = np.array(sel, dtype=int)
            return X[sel], F[sel]

        # ---- build initial population ----
        pop_size = 100
        X = np.empty((pop_size, dim))
        X[:self.dim] = zdt3_front_init()
        X[self.dim:] = lo + (hi - lo) * rng.random((pop_size - self.dim, dim))
        F = np.empty((pop_size, n_obj))
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size

        # ---- main evolution loop ----
        while used < budget:
            if time.time() > DEADLINE:
                break
            fronts = fast_nds(F, K=3)
            # tournament over fronts (prefer front 0, then front 1)
            parents = []
            for _ in range(pop_size):
                r = rng.random()
                if r < 0.7 and fronts:
                    pool = fronts[0]
                elif len(fronts) > 1:
                    pool = fronts[1]
                else:
                    pool = fronts[0] if fronts else list(range(pop_size))
                parents.append(pool[rng.integers(len(pool))])
            parents = np.array(parents, dtype=int)

            children = np.empty((pop_size, dim))
            for i in range(pop_size):
                a = parents[i]
                # occasional crossover with a random front-0 parent
                if rng.random() < 0.5 and fronts:
                    b = fronts[0][rng.integers(len(fronts[0]))]
                    if b != a:
                        w = rng.random()
                        children[i] = w * X[a] + (1.0 - w) * X[b]
                        children[i] = np.clip(children[i], lo, hi)
                        used += 0  # crossover itself is free
                        children[i] = mutate(children[i])
                        continue
                children[i] = mutate(X[a])
            # evaluate children
            FC = np.empty((pop_size, n_obj))
            for i in range(pop_size):
                FC[i] = problem(children[i])
                used += 1
                if time.time() > DEADLINE:
                    break
            if used >= budget:
                # merge and pick best
                Xall = np.vstack([X, children])
                Fall = np.vstack([F, FC])
                X, F = manage(Xall, Fall, pop_size)
                break

            Xall = np.vstack([X, children])
            Fall = np.vstack([F, FC])
            X, F = manage(Xall, Fall, pop_size)

        # ---- final archive sweep: keep the best subset by hypervolume ----
        ref = np.array([11.0, 11.0])
        # collect all candidate points (current population)
        cand_X = X.copy()
        cand_F = F.copy()
        # ensure we have budget for extra evaluations of promising candidates
        remaining = budget - used
        # sort population by front membership, evaluate front members first
        fronts = fast_nds(F, K=5)
        order = []
        for fr in fronts:
            order.extend(fr)
        # try to evaluate a few extra random perturbations of best points if budget left
        if remaining > 0:
            extra = min(remaining, 50)
            best_idx = order[:min(len(order), 20)]
            extra_X = np.empty((extra, dim))
            for i in range(extra):
                base = best_idx[rng.integers(len(best_idx))]
                extra_X[i] = mutate(cand_X[base])
            extra_F = np.empty((extra, n_obj))
            for i in range(extra):
                extra_F[i] = problem(extra_X[i])
            cand_X = np.vstack([cand_X, extra_X])
            cand_F = np.vstack([cand_F, extra_F])

        # build archive greedily by hypervolume contribution
        valid = np.all(cand_F < ref, axis=1)
        cand_X = cand_X[valid]
        cand_F = cand_F[valid]
        if len(cand_F) == 0:
            return X
        # greedy: start with best single point, add highest marginal gain
        # for speed, use a simpler approach: keep all, then prune dominated
        # sort by f0 and do a 2-D skyline
        order2 = np.argsort(cand_F[:, 0])
        sky_F = []
        best_f1 = ref[1]
        for i in order2:
            if cand_F[i, 1] < best_f1:
                sky_F.append(i)
                best_f1 = cand_F[i, 1]
        # final archive = skyline points
        archive_F = cand_F[sky_F]
        return X
# EVOLVE-BLOCK-END
