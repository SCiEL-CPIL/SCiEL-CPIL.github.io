"""Optimized program: two interlocking regular tetrahedra + 6 extra points,
differential evolution + local polish, with deadline guard."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14
D = 3
DEADLINE = 130.0  # safety margin before 150s kill


def _pairwise_dists_sq(pts):
    """Return (n, n) matrix of squared pairwise distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _score(pts):
    """Squared ratio of max to min pairwise distance (0 on diagonal)."""
    d2 = _pairwise_dists_sq(pts)
    np.fill_diagonal(d2, np.inf)
    dmin2 = d2.min()
    dmax2 = d2.max()
    if dmin2 <= 0:
        return np.inf
    return dmax2 / dmin2


def _regular_tetrahedron(scale=1.0):
    """4 vertices of a regular tetrahedron centered at origin."""
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) * (scale / np.sqrt(3.0))
    return v


def _build_from_params(p):
    """
    p = [s1, rx1, ry1, rz1, s2, rx2, ry2, rz2, tx, ty, tz,
         e0x, e0y, e0z, e1x, e1y, e1z, e2x, e2y, e2z,
         e3x, e3y, e3z, e4x, e4y, e4z, e5x, e5y, e5z]
    Two tetrahedra (each scaled+rotated) + 6 free extra points.
    """
    s1, rx1, ry1, rz1 = p[0], p[1], p[2], p[3]
    s2, rx2, ry2, rz2 = p[4], p[5], p[6], p[7]
    tx, ty, tz = p[8], p[9], p[10]
    extras = p[11:].reshape(6, 3)

    def rot(rx, ry, rz):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    t1 = _regular_tetrahedron(s1) @ rot(rx1, ry1, rz1).T
    t2 = _regular_tetrahedron(s2) @ rot(rx2, ry2, rz2).T
    t2 = t2 + np.array([tx, ty, tz])

    pts = np.vstack([t1, t2, extras])
    # Center to reduce drift
    pts = pts - pts.mean(axis=0)
    return pts


def _objective(p):
    pts = _build_from_params(p)
    return _score(pts)


def construct():
    t0 = time.time()

    def remaining():
        return DEADLINE - (time.time() - t0)

    # Bounds: scales, rotations, translation, extras
    bounds = (
        [(0.3, 2.5),] + [(-np.pi, np.pi),] * 3 +  # tet1
        [(0.3, 2.5),] + [(-np.pi, np.pi),] * 3 +  # tet2
        [(-1.5, 1.5),] * 3 +  # translation
        [(-2.5, 2.5),] * 18  # 6 extras
    )

    best_pts = None
    best_score = np.inf

    def update_best(pts, sc):
        nonlocal best_pts, best_score
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()

    # Symmetric starting guess: stella octangula + 6 points on a ring
    s0 = 1.0
    t1_0 = _regular_tetrahedron(s0)
    # Second tetrahedron rotated 180 about x-axis (dual)
    Rx180 = np.diag([1.0, -1.0, -1.0])
    t2_0 = _regular_tetrahedron(s0) @ Rx180.T
    # 6 points: two triangles (top and bottom) at radius r
    r = s0 * np.sqrt(3.0) / np.sqrt(2.0)
    z_off = s0 * 0.5
    top = np.array([[r * np.cos(2 * np.pi * k / 3), r * np.sin(2 * np.pi * k / 3), z_off] for k in range(3)])
    bot = np.array([[r * np.cos(2 * np.pi * k / 3 + np.pi / 3), r * np.sin(2 * np.pi * k / 3 + np.pi / 3), -z_off] for k in range(3)])
    extras0 = np.vstack([top, bot])
    pts0 = np.vstack([t1_0, t2_0, extras0])
    pts0 = pts0 - pts0.mean(axis=0)
    update_best(pts0, _score(pts0))

    # Differential evolution
    if remaining() > 20:
        try:
            res = differential_evolution(
                _objective, bounds,
                seed=42,
                maxiter=40,
                popsize=12,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol',
            )
            if res.success or res.fun < best_score:
                p = res.x
                pts = _build_from_params(p)
                update_best(pts, res.fun)
        except Exception:
            pass

    # Local polish on best params
    if best_pts is not None and remaining() > 5:
        # Reconstruct params from best_pts is hard; instead polish in coordinate space
        try:
            def coord_obj(x):
                pts = x.reshape(N, D)
                return _score(pts)

            res2 = minimize(
                coord_obj, best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8},
            )
            if res2.fun < best_score:
                update_best(res2.x.reshape(N, D), res2.fun)
        except Exception:
            pass

    # Final deadline check
    if best_pts is None:
        # Fallback to initial symmetric guess
        best_pts = pts0

    return best_pts


if __name__ == '__main__':
    pts = construct()
    print("Score:", _score(pts))
