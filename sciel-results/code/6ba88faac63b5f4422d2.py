# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting basin identification for BBOB f4 (Büche-Rastrigin).

    Büche-Rastrigin has a periodic ridge structure: local minima sit on a lattice
    with period P = 2*sqrt(2) per axis, and the global minimum is at the (shifted)
    origin where the quadratic basin term is smallest. We (1) sample a coarse grid
    to map the landscape, (2) generate candidate lattice points using the known
    period, and (3) apply a fast line-search / coordinate refinement to the best
    candidates, all under an explicit wall-clock deadline guard.
    """

    P = 2.0 * np.sqrt(2.0)          # ridge period of Büche-Rastrigin
    DEADLINE = 290.0                 # safety margin below the 300s limit

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._t0 = None
        self._calls = 0
        self._best_x = None
        self._best_f = None

    # ---- bookkeeping -------------------------------------------------------
    def _eval(self, func, x):
        x = np.asarray(x, dtype=float)
        f = float(func(x))
        self._calls += 1
        if not np.isfinite(f):
            f = 1e18
        if self._best_f is None or f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()
        return f

    def _time_ok(self):
        return (time.time() - self._t0) < self.DEADLINE

    def _budget_ok(self):
        return self._calls < self.budget

    # ---- fast refinement: coordinate line search ---------------------------
    def _refine(self, func, x, fx, max_iter=40):
        """Coordinate-wise line search with adaptive shrinking step.

        Near a local minimum the function is smooth and convex-ish, so a
        coordinate descent with a shrinking step converges to high precision
        in few iterations. Returns (x, fx).
        """
        x = x.copy()
        step = 1.0
        for _ in range(max_iter):
            if not (self._time_ok() and self._budget_ok()):
                break
            improved = False
            for i in range(self.dim):
                # line search along axis i: try +step, -step, halve if both fail
                s = step
                best_local = (fx, x.copy())
                for _ in range(4):
                    if not (self._time_ok() and self._budget_ok()):
                        break
                    y = x.copy()
                    y[i] += s
                    y = np.clip(y, -5, 5)
                    fy = self._eval(func, y)
                    if fy < best_local[0]:
                        best_local = (fy, y)
                        improved = True
                    y = x.copy()
                    y[i] -= s
                    y = np.clip(y, -5, 5)
                    fy = self._eval(func, y)
                    if fy < best_local[0]:
                        best_local = (fy, y)
                        improved = True
                    s *= 0.5
                if best_local[0] < fx:
                    x, fx = best_local
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        return x, fx

    def __call__(self, func):
        self._t0 = time.time()
        self._calls = 0
        self._best_x = None
        self._best_f = None
        dim = self.dim
        P = self.P

        # ---- Phase 1: coarse grid to map the landscape ---------------------
        # 3 points per axis (low, mid, high) -> 3^5 = 243 points.
        grid_vals = np.array([-4.0, 0.0, 4.0])
        candidates = []
        for a in grid_vals:
            for b in grid_vals:
                for c in grid_vals:
                    for d in grid_vals:
                        for e in grid_vals:
                            if not (self._time_ok() and self._budget_ok()):
                                break
                            p = np.array([a, b, c, d, e])
                            f = self._eval(func, p)
                            candidates.append((f, p))

        # ---- Phase 2: lattice candidates from periodicity -------------------
        # Local minima of the periodic part sit at x0 + k*P. The quadratic term
        # favors points near the origin, so we enumerate lattice points near 0.
        ks = [-2, -1, 0, 1, 2]
        lattice_scores = []
        seen = set()
        for k0 in ks:
            for k1 in ks:
                for k2 in ks:
                    for k3 in ks:
                        for k4 in ks:
                            if not (self._time_ok() and self._budget_ok()):
                                break
                            p = np.array([k0, k1, k2, k3, k4], dtype=float) * P
                            if np.any(p < -5) or np.any(p > 5):
                                continue
                            key = tuple(np.round(p, 6))
                            if key in seen:
                                continue
                            seen.add(key)
                            f = self._eval(func, p)
                            lattice_scores.append((f, p))

        # ---- Phase 3: refine the best candidates ---------------------------
        all_cands = candidates + lattice_scores
        all_cands.sort(key=lambda t: t[0])

        top_n = 8
        for f, p in all_cands[:top_n]:
            if not (self._time_ok() and self._budget_ok()):
                break
            x, fx = self._refine(func, p, f, max_iter=40)

        # ---- Phase 4: final polish on the best point -----------------------
        if self._best_x is not None and self._time_ok() and self._budget_ok():
            x, fx = self._refine(func, self._best_x, self._best_f, max_iter=60)

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
