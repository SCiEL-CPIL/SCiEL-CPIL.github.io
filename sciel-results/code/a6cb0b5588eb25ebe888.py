"""Hybrid grid-subdivision + quadratic-surrogate optimizer for BBOB Rastrigin (f3).

Strategy (combines two approaches into one coherent design):
  Phase 1 (exploration): deterministic hierarchical 2^dim grid subdivision.
      Start on [-5,5]^dim, evaluate the 2^dim cell centers, keep the best cell,
      subdivide. This is memoryless, cheap (32 evals/level for dim=5) and
      robustly lands inside the basin of the hidden minimum.
  Phase 2 (exploitation): once the search region is small enough that we are
      almost certainly inside a single basin, stop the pure grid and switch to
      a local quadratic surrogate (ridge-regularised least squares over the
      points we have already collected) plus an adaptive line search. This
      reuses the grid's own evaluations as training data, so the refinement
      machinery is "free" relative to the exploration cost.

The grid gives robust global coverage; the surrogate gives fast local
convergence. Neither alone is as effective as the combination.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin well under 300s

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        points = []  # list of (x, f) for the surrogate model

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            points.append((x.copy(), fx))
            return fx

        # ------------------------------------------------------------------
        # Phase 1: hierarchical grid subdivision
        # ------------------------------------------------------------------
        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)
        n_per_dim = 2
        n_total = n_per_dim ** dim

        # Region size at which we switch to local exploitation.
        # For dim=5, a side of ~0.5 is comfortably inside one basin of the
        # period-2 Rastrigin landscape.
        switch_side = 0.5

        while calls_made < budget and time.time() < deadline:
            region_side = float(np.max(high - low))
            if region_side <= switch_side:
                break  # switch to exploitation

            if calls_made + n_total > budget:
                # Can't afford a full level; evaluate what we can and stop.
                for idx in range(n_total):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    x = np.zeros(dim)
                    rem = idx
                    for d in range(dim):
                        pos = rem % n_per_dim
                        rem //= n_per_dim
                        cw = (high[d] - low[d]) / n_per_dim
                        x[d] = low[d] + (pos + 0.5) * cw
                    counted_func(x)
                break

            best_in_level = np.inf
            best_x_in_level = None
            for idx in range(n_total):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                rem = idx
                for d in range(dim):
                    pos = rem % n_per_dim
                    rem //= n_per_dim
                    cw = (high[d] - low[d]) / n_per_dim
                    x[d] = low[d] + (pos + 0.5) * cw
                fx = counted_func(x)
                if fx < best_in_level:
                    best_in_level = fx
                    best_x_in_level = x.copy()

            if best_x_in_level is None:
                break

            # Zoom into the best cell.
            new_low = np.zeros(dim)
            new_high = np.zeros(dim)
            for d in range(dim):
                cw = (high[d] - low[d]) / n_per_dim
                pos = int((best_x_in_level[d] - low[d]) / cw)
                pos = max(0, min(n_per_dim - 1, pos))
                new_low[d] = low[d] + pos * cw
                new_high[d] = low[d] + (pos + 1) * cw
            low, high = new_low, new_high

            if np.all(high - low < 1e-6):
                break

        # ------------------------------------------------------------------
        # Phase 2: local quadratic surrogate + adaptive line search
        # ------------------------------------------------------------------
        if calls_made < budget:
            self._exploit(func, counted_func, deadline)

        return best_x, best_f

    # ------------------------------------------------------------------
    def _exploit(self, func, counted_func, deadline):
        dim = self.dim
        budget = self.budget

        def calls_left():
            return budget - counted_func.__self_calls if False else (budget - self._calls())

        # We track calls via a mutable holder.
        # (counted_func already increments the outer counter; expose it here.)

        # Build a quadratic model from the points collected so far.
        def build_model(center, radius):
            pts = points
            if len(pts) < 3:
                return None
            # Use points within radius of center, nearest-first, up to ~20.
            idx = list(range(len(pts)))
            idx.sort(key=lambda i: np.linalg.norm(pts[i][0] - center))
            sel = []
            for i in idx:
                if np.linalg.norm(pts[i][0] - center) <= radius:
                    sel.append(i)
                if len(sel) >= 20:
                    break
            if len(sel) < 3:
                sel = idx[:min(20, len(idx))]
            if len(sel) < 3:
                return None
            X = np.zeros((len(sel), 1 + dim + dim * (dim + 1) // 2))
            y = np.zeros(len(sel))
            for r, i in enumerate(sel):
                xv, fv = pts[i]
                v = xv - center
                X[r, 0] = 1.0
                X[r, 1:1 + dim] = v
                q = 1 + dim
                for a in range(dim):
                    for b in range(a, dim):
                        X[r, q] = v[a] * v[b]
                        q += 1
                y[r] = fv
            lam = 1e-3
            try:
                w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
            except np.linalg.LinAlgError:
                return None
            return w, center

        def model_min(w, center):
            b = w[1:1 + dim]
            A = np.zeros((dim, dim))
            q = 1 + dim
            for a in range(dim):
                for b2 in range(a, dim):
                    val = w[q]
                    if a == b2:
                        A[a, a] += val
                    else:
                        A[a, b2] += val / 2.0
                        A[b2, a] += val / 2.0
                    q += 1
            # Minimize b^T v + v^T A v  ->  2 A v = -b
            try:
                if np.linalg.cond(A) < 1e6:
                    v_star = -0.5 * np.linalg.solve(A, b)
                    return center + v_star
            except np.linalg.LinAlgError:
                pass
            return center

        # Local adaptive line search: try candidate steps, keep the best.
        # Directions: axis-aligned + a few random unit vectors.
        rng = np.random.default_rng(7)
        dirs = [np.eye(dim, dim=dim)[d] for d in range(dim)]
        for _ in range(8):
            v = rng.normal(size=dim)
            n = np.linalg.norm(v)
            if n > 1e-10:
                dirs.append(v / n)

        step = 0.25
        no_improve = 0
        last_best = best_f

        while calls_left() > 0 and time.time() < deadline:
            # Rebuild the surrogate periodically.
            if no_improve >= 4 or no_improve == 0:
                w, center = build_model(best_x, step * 3.0 + 0.5)
                if w is not None:
                    cand = model_min(w, center)
                    cand = np.clip(cand, -5.0, 5.0)
                    if np.linalg.norm(cand - best_x) > 1e-8:
                        fx = counted_func(cand)
                        if fx < last_best - 1e-8:
                            last_best = best_f
                            no_improve = 0
                            step = min(step * 1.5, 1.0)
                            continue
                        else:
                            no_improve += 1

            # Line search along a direction.
            d = dirs[no_improve % len(dirs)]
            improved = False
            for sign in (1.0, -1.0):
                cand = np.clip(best_x + sign * step * d, -5.0, 5.0)
                fx = counted_func(cand)
                if fx < last_best - 1e-8:
                    last_best = best_f
                    no_improve = 0
                    improved = True
                    break
            if improved:
                step = min(step * 1.3, 1.0)
            else:
                no_improve += 1
                step = max(step * 0.6, 1e-4)
                if step < 1e-4:
                    break

    # helper to read the call counter from the closure
    def _calls(self):
        # The outer closure stores calls_made; expose via attribute.
        return getattr(self, "_call_counter", 0)
