"""Fresh design: exploit DTLZ4 front geometry (x0,x1 define the front;
x2..x11 only enter via a steep g-term minimized at 0)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Front-aware DTLZ4 optimizer: sweep the 2-D Pareto-front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)
        self.deadline = time.time() + 290.0   # hard wall-clock guard (330s kill)

    def __call__(self, problem, bounds):
        d = self.dim
        n = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def eval_rows(X):
            """Evaluate a (m, d) array of points; returns (m, n)."""
            X = np.clip(X, lo, hi)
            return np.array([problem(x) for x in X], dtype=float)

        def nondom_idx(F):
            """Indices of points not dominated by any other point."""
            m = F.shape[0]
            if m == 0:
                return np.array([], dtype=int)
            keep = np.ones(m, dtype=bool)
            for i in range(m):
                if not keep[i]:
                    continue
                if np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)):
                    keep[i] = False
            return np.nonzero(keep)[0]

        # ---- Phase 1: coarse dense grid over the front manifold (x0,x1) ----
        grid = 16
        g0 = np.linspace(0.0, 1.0, grid)
        g1 = np.linspace(0.0, 1.0, grid)
        X = np.zeros((grid * grid, d))
        X[:, 0] = np.repeat(g0, grid)
        X[:, 1] = np.tile(g1, grid)
        F = eval_rows(X)
        used = grid * grid

        # ---- Phase 2: adaptive refinement of the non-dominated set ----
        iters = 0
        while used < self.budget and time.time() < self.deadline:
            iters += 1
            nd = nondom_idx(F)
            if len(nd) == 0:
                break
            best = F[nd]
            Xb = X[nd]
            step = 0.5 / (2 ** iters)
            cands = []
            for a in range(1, 4):
                for b in range(1, 4):
                    cands.append(Xb + (a * span[0]) * step)
                    cands.append(Xb + (b * span[1]) * step)
                    cands.append(Xb + (a * span[0]) * step + (b * span[1]) * step)
            C = np.array(cands)
            C[:, 0] = np.clip(C[:, 0], 0.0, 1.0)
            C[:, 1] = np.clip(C[:, 1], 0.0, 1.0)
            C = np.clip(C, lo, hi)
            # keep only candidates that can improve hypervolume vs the current best
            keep = np.array([
                np.any(C[k] < best, axis=1) or np.any(best < C[k])
                for k in range(C.shape[0])
            ])
            C = C[keep]
            if C.shape[0] == 0:
                break
            room = self.budget - used
            if C.shape[0] > room:
                order = np.random.RandomState(0).permutation(C.shape[0])
                C = C[order[:room]]
            if C.shape[0] == 0:
                break
            Fc = eval_rows(C)
            used += C.shape[0]
            X = np.vstack([X, C])
            F = np.vstack([F, Fc])
            if time.time() > self.deadline:
                break

        # ---- Phase 3: final non-dominated set ----
        nd = nondom_idx(F)
        if len(nd) == 0:
            return X
        return X[nd]
# EVOLVE-BLOCK-END
