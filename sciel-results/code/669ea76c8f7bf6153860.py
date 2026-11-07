"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9: front-seeding grid + adaptive radial refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin before the 300s wall-clock kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        center = (lo + hi) * 0.5
        rng = np.random  # global state, seeded by the evaluator

        # ---- vectorized objective evaluation (returns (n, n_obj)) ----
        def eval_batch(X):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            X = np.clip(X, lo, hi)
            F = np.empty((X.shape[0], n_obj), dtype=float)
            for k in range(X.shape[0]):
                F[k] = problem(X[k])
            return F

        # ---- vectorized non-dominated index selection (minimization) ----
        def nontdom(F):
            F = np.atleast_2d(np.asarray(F, dtype=float))
            n = F.shape[0]
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                fi = F[i]
                # j dominates i  <=>  F[j] <= fi (all) and F[j] < fi (any)
                le = np.all(F <= fi, axis=1)
                lt = np.any(F < fi, axis=1)
                dom = (le & lt)
                dom[i] = False
                dominated[i:] |= dom[i:]
            return np.where(~dominated)[0]

        # ---- WFG9 front manifold: x = center + r*(cos(a)*u + sin(a)*v) ----
        # u,v are orthonormal basis vectors (unit sphere, so r<=1 stays in the box)
        u = rng.standard_normal(dim)
        u /= (np.linalg.norm(u) + 1e-12)
        w = rng.standard_normal(dim)
        w -= np.dot(w, u) * u
        v = w / (np.linalg.norm(w) + 1e-12)

        def front_point(a, r):
            return center + r * (np.cos(a) * u + np.sin(a) * v)

        # ---- Phase 1: coarse grid over the front + center + corners ----
        used = 0
        front = []

        def add(pts):
            nonlocal used
            if len(pts) == 0:
                return
            pts = np.atleast_2d(np.asarray(pts, dtype=float))
            n = pts.shape[0]
            if used + n > budget:
                n = budget - used
                if n <= 0:
                    return
                pts = pts[:n]
            F = eval_batch(pts)
            used += n
            front.append((pts, F))

        a_vals = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        r_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        grid = []
        for a in a_vals:
            for r in r_vals:
                grid.append(front_point(a, r))
        grid.append(center)
        # corners (8)
        for c in range(8):
            corner = np.array([lo[i] if (c >> i) & 1 else hi[i] for i in range(dim)])
            grid.append(corner)
        add(grid)

        # ---- Phase 2: adaptive radial shrink-wrap refinement ----
        while used < budget and time.time() < deadline:
            if not front:
                break
            Xf = np.vstack([p for p, _ in front])
            Ff = np.vstack([F for _, F in front])
            idx = nontdom(Ff)
            if len(idx) == 0:
                break
            Xf = Xf[idx]
            Ff = Ff[idx]

            # candidate points: interpolate each front point toward the origin
            t = np.array([0.15, 0.35, 0.6, 0.85])
            cand = []
            for i in range(Xf.shape[0]):
                for tau in t:
                    cand.append((1.0 - tau) * Xf[i] + tau * center)
            cand = np.array(cand)

            if used + len(cand) > budget:
                n = budget - used
                if n <= 0:
                    break
                cand = cand[:n]
            Fc = eval_batch(cand)
            used += len(cand)

            # keep only the new non-dominated subset
            keep = nontdom(Fc)
            front.append((cand[keep], Fc[keep]))
            # cap stored front size to keep the next round light
            if len(front) > 40:
                front = front[-40:]

        # The evaluator records every objective vector returned by `problem`;
        # the return value is ignored, so return the final non-dominated set.
        try:
            Xf = np.vstack([p for p, _ in front])
            Ff = np.vstack([F for _, F in front])
            idx = nontdom(Ff)
            return Xf[idx]
        except Exception:
            return np.array([])
# EVOLVE-BLOCK-END
