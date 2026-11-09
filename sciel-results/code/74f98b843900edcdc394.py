# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon (vertices in CCW order) that maximizes the minimum per-vertex
    equidistance score.

    Design (from scratch, global search):
      * Represent the polygon by 20 strictly-increasing polar angles on [0, 2*pi) at a
        fixed radius.  This parameterization guarantees strict convexity (no repeated
        points, all turns the same sign, total turning 2*pi), so the result is always a
        valid convex polygon.
      * Score is scale/translation invariant, so we work in the normalized frame
        (centroid subtracted, divided by mean radius) to match the evaluator exactly.
      * Optimize the 20 angles with differential_evolution (population-based global
        search) which explores the continuous angular landscape better than random
        restarts / greedy hill-climbing.
      * A wall-clock deadline guard returns the best polygon found with a safety margin.

    Returns:
        np.ndarray of shape (20, 2), finite, in cyclic (CCW) order.
    """
    n = 20
    start = time.time()
    deadline = 540.0          # safety margin well under the 630 s kill
    radius = 1.0

    # ---------- scoring helpers (match the public contract exactly) ----------
    def _normalize(pts):
        c = pts.mean(axis=0)
        d = pts - c
        mean_r = np.mean(np.linalg.norm(d, axis=1))
        if mean_r <= 0.01:
            return None
        return d / mean_r

    def _vertex_score(dists):
        # dists: the 19 distances from this vertex to the others, already sorted
        if np.any(dists < 0.05):
            return 0.0
        diffs = np.diff(dists)
        k = int(np.argmin(diffs))
        D = 0.5 * (dists[k] + dists[k + 1])
        if D < 0.1:
            return 0.0
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k - 1, k, k + 1, k + 2]
        sel = dists[idx]
        raw = np.mean(np.maximum(sel / D, D / sel))
        if raw <= 0.0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _polygon_score(pts):
        p = _normalize(pts)
        if p is None:
            return 0.0
        m = p.shape[0]
        # pairwise distances
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        best = 1.0
        for i in range(m):
            other = np.delete(dist[i], i)
            other.sort()
            s = _vertex_score(other)
            if s < best:
                best = s
                if best <= 0.0:
                    break
        return best

    def _obj(angles):
        a = np.asarray(angles, dtype=float)
        a = np.sort(a)
        a = (a - a[0]) % (2.0 * np.pi)          # wrap to [0, 2*pi)
        x = radius * np.cos(a)
        y = radius * np.sin(a)
        return -_polygon_score(np.stack([x, y], axis=1))

    # ---------- baseline: regular polygon ----------
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base_pts = np.stack([radius * np.cos(base_angles), radius * np.sin(base_angles)], axis=1)
    best = base_pts.copy()
    best_score = _polygon_score(base_pts)

    # ---------- global search: differential evolution over angles ----------
    bounds = [(0.0, 2.0 * np.pi) for _ in range(n)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=20240611,
            maxiter=1200,
            popsize=22,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.8,
            polish=False,
            init="sobol",
            callback=lambda xk, convergence: (
                True if (time.time() - start) < deadline else False
            ),
        )
        if res is not None and res.x is not None:
            cand = res.x
            cand = np.sort(cand)
            cand = (cand - cand[0]) % (2.0 * np.pi)
            pts = np.stack([radius * np.cos(cand), radius * np.sin(cand)], axis=1)
            sc = _polygon_score(pts)
            if sc > best_score:
                best, best_score = pts, sc
    except Exception:
        pass

    # ---------- local refinement (Nelder-Mead) on the best angles, if time remains ----------
    if (time.time() - start) < deadline - 60.0:
        try:
            from scipy.optimize import minimize
            a0 = np.sort(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))
            if best is not base_pts:
                # recover angles of the best polygon for a warm start
                a0 = np.arctan2(best[:, 1], best[:, 0])
                a0 = (a0 - a0.min()) % (2.0 * np.pi)
                a0 = np.sort(a0)

            def _wrap(a):
                a = np.sort(a)
                return (a - a[0]) % (2.0 * np.pi)

            def _obj2(a):
                return _obj(_wrap(a))

            # bounded-ish refinement via penalty-free Nelder-Mead on wrapped angles
            r2 = minimize(_obj2, a0, method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
            if r2 is not None and r2.x is not None:
                cand = _wrap(r2.x)
                pts = np.stack([radius * np.cos(cand), radius * np.sin(cand)], axis=1)
                sc = _polygon_score(pts)
                if sc > best_score:
                    best, best_score = pts, sc
        except Exception:
            pass

    # ---------- final safety: guarantee a valid convex polygon ----------
    if _polygon_score(best) <= 0.0:
        best = base_pts
    return best


# EVOLVE-BLOCK-END
