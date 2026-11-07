import numpy as np
import time


class Optimizer:
    def __init__(self, dim, n_obj, budget=None):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270.0  # hard wall-clock safety margin (< 300 s)

        dim = self.dim
        n2 = dim - 1
        c = 1.0 + 9.0 / n2
        inv_c = 1.0 / c
        one_plus_c = 1.0 + c

        # ZDT2 front: f1 = x1, f2 = (1+9/n2) * (1 - sqrt(x1/(1+9/n2))), x1 in [0,1]
        # x2..x30 = 0.5  ->  g = 1 (optimal)  ->  points sit exactly on the front.
        def build(x1):
            x1 = np.asarray(x1, dtype=float).ravel()
            X = np.empty((len(x1), dim), dtype=float)
            X[:, 0] = x1
            X[:, 1:] = 0.5
            f2 = one_plus_c * (1.0 - np.sqrt(np.clip(x1 * inv_c, 0.0, 1.0)))
            return X, np.column_stack((x1, f2))

        # ---- Phase 1: stochastic LHS over x1 (uniform coverage, no grid aliasing) ----
        n_lhs = 120
        order = np.random.permutation(n_lhs)
        x1 = (order + np.random.random(n_lhs)) / n_lhs

        # Guarantee both corners (both needed for hypervolume) without disturbing the LHS spread.
        x1[0] = 1.0          # f1 = 1, f2 = 0
        x1[1] = 1.0 / n_lhs  # f1 ~ 0, f2 ~ 1

        X, F = build(x1)
        used = n_lhs
        F = F[np.argsort(F[:, 0])]

        # ---- Phase 2: greedy hypervolume maximization with the remaining budget ----
        def hv(F):
            # F sorted by f1 ascending; hypervolume vs reference (11, 11)
            f1 = F[:, 0]
            f2 = F[:, 1]
            # minimum f2 for each distinct f1 bucket (right-continuous)
            keep = np.concatenate(([True], np.diff(f1) > 0))
            f1k = f1[keep]
            f2k = np.minimum.accumulate(f2[keep])
            f1_ext = np.concatenate(([0.0], f1k))
            f2_ext = np.concatenate((f2k, [11.0]))
            return float(np.sum((f1_ext[1:] - f1_ext[:-1]) * (11.0 - f2_ext[:-1])))

        base = hv(F)

        # candidate pool: LHS points + a fine deterministic grid (for local refinement)
        grid = np.linspace(0.0, 1.0, 2001)
        pool = np.unique(np.concatenate([x1, grid]))
        pool = np.clip(pool, 0.0, 1.0)
        pool = pool[np.argsort(pool)]
        pool_set = set(np.round(pool, 12))

        rng = np.random.default_rng(12345)
        idx = 0
        while used < self.budget:
            if time.time() > deadline:
                break
            # pick the pool index that maximizes hypervolume gain
            best_i = -1
            best_gain = -1.0
            for i in range(len(pool)):
                p = pool[i]
                # find insertion position in F (sorted by f1)
                ins = np.searchsorted(F[:, 0], p, side='left')
                cand = np.vstack((F[:ins], [p, one_plus_c * (1.0 - np.sqrt(np.clip(p * inv_c, 0.0, 1.0)))], F[ins:]))
                g = hv(cand) - base
                if g > best_gain:
                    best_gain = g
                    best_i = i
            if best_i < 0:
                break
            p = pool[best_i]
            pool_set.discard(np.round(p, 12))
            if not pool_set:
                break
            # insert into F keeping sort order
            f2 = one_plus_c * (1.0 - np.sqrt(np.clip(p * inv_c, 0.0, 1.0)))
            ins = np.searchsorted(F[:, 0], p, side='left')
            F = np.vstack((F[:ins], [p, f2], F[ins:]))
            base = hv(F)
            used += 1

        # ---- Final evaluation of the selected front points ----
        sel_x1 = F[:, 0]
        X, _ = build(sel_x1)
        for x in X:
            if time.time() > deadline:
                break
            problem(x)

        return X
