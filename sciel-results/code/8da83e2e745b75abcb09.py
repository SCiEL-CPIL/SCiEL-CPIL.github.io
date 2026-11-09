# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Fast multi-objective evolutionary search for WFG9 (3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 290.0
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng_span = (hi - lo) / (2.0 * np.sqrt(self.dim))

        def fast_non_dom(F, n):
            """Return indices of non-dominated points in F (n x m)."""
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                f = F[i]
                beater = np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
                beater[i] = False
                if beater.any():
                    keep[i] = False
                    keep[beater] = False
            return np.where(keep)[0]

        def eval_block(X):
            out = np.empty((X.shape[0], self.n_obj), dtype=float)
            for i in range(X.shape[0]):
                out[i] = problem(X[i])
            return out

        # --- Initial population: random + corner (extreme) points ---
        init_pop = int(min(self.budget // 3, 100))
        X = lo + (hi - lo) * rng.random((init_pop, self.dim))
        n_obj = self.n_obj
        corners = np.empty((n_obj * 2, self.dim), dtype=float)
        for k in range(n_obj):
            corners[2 * k] = lo
            corners[2 * k][k] = hi[k]
            corners[2 * k + 1] = hi
            corners[2 * k + 1][k] = lo[k]
        X = np.vstack([X, corners])
        F = eval_block(X)
        used = X.shape[0]
        if used > self.budget:
            idx = np.random.default_rng(0).choice(used, self.budget, replace=False)
            X, F = X[idx], F[idx]
            used = self.budget

        # --- Main loop ---
        while used < self.budget:
            if time.time() > deadline:
                break
            nd = fast_non_dom(F, F.shape[0])
            if len(nd) < 2:
                nd = np.arange(F.shape[0])
            parents = X[nd]
            pf = F[nd]
            n_par = len(parents)

            # adaptive mutation scale
            base = 0.15
            spread = (np.max(pf, axis=0) - np.min(pf, axis=0))
            scale = max(0.05, min(0.4, base + 0.25 * np.mean(spread)))

            # batch size: respect budget and deadline
            room = self.budget - used
            batch = min(room, max(1, room // 2))

            # choose parents (random) and mutation strengths (1-2 parents)
            pidx = rng.integers(0, n_par, size=batch)
            npar = rng.integers(1, 3, size=batch)
            children = np.empty((batch, self.dim), dtype=float)
            for b in range(batch):
                pi = pidx[b]
                if npar[b] == 1:
                    c = parents[pi].copy()
                else:
                    j = rng.integers(0, n_par)
                    c = 0.5 * (parents[pi] + parents[j])
                c += scale * rng_span * rng.standard_normal(self.dim)
                children[b] = c
            children = np.clip(children, lo, hi)

            fc = eval_block(children)
            used += batch

            # replace worst (dominated) points with new children
            dom = np.ones(F.shape[0], dtype=bool)
            dom[nd] = False
            if not dom.any():
                dom = np.ones(F.shape[0], dtype=bool)
                dom[nd[:max(1, len(nd) // 2)]] = False
            widx = np.where(dom)[0]
            if len(widx) == 0:
                continue
            n_replace = min(batch, len(widx))
            rep = rng.choice(widx, size=n_replace, replace=False)
            X[rep] = children[:n_replace]
            F[rep] = fc[:n_replace]

        return X
# EVOLVE-BLOCK-END
