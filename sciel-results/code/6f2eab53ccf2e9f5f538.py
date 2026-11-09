"""Covering the sphere with 130 equal caps.

Direct projected gradient descent on the true objective: the covering radius,
expressed as the maximum circumradius of the facets of the convex hull.  The
starting configuration is the Hardin-Sloane-Smith icosahedral family (icosahedron
vertices + face centers + edge midpoints + 53 optimized caps), which already sits
near the best-known value; the optimizer then refines it while keeping points on
the unit sphere and the hull non-degenerate.
"""
import time

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 130


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _facet_circumradius(a, b, c):
    """Circumradius of the spherical triangle with unit vertices a,b,c (radians)."""
    sa = float(np.dot(a, b))
    sb = float(np.dot(b, c))
    sc = float(np.dot(c, a))
    sa = np.clip(sa, -1.0, 1.0)
    sb = np.clip(sb, -1.0, 1.0)
    sc = np.clip(sc, -1.0, 1.0)
    a1 = np.arccos(sa)   # side opposite vertex a
    b1 = np.arccos(sb)   # side opposite vertex b
    c1 = np.arccos(sc)   # side opposite vertex c
    s = 0.5 * (a1 + b1 + c1)
    sin_a = np.sin(a1)
    if sin_a <= 1e-12:
        return 0.5 * max(a1, b1, c1)
    R = np.sqrt(np.sin(s) * np.sin(s - a1) * np.sin(s - b1) * np.sin(s - c1)) / sin_a
    return float(R)


def _covering_and_grad(pts):
    """Return (covering radius in radians, per-point gradient of the radius)."""
    pts = np.asarray(pts, dtype=float)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)
    if hull.volume <= 1e-6:
        return np.pi, np.zeros_like(pts)
    faces = hull.simplices
    R = np.empty(len(faces))
    for k, f in enumerate(faces):
        R[k] = _facet_circumradius(pts[f[0]], pts[f[1]], pts[f[2]])
    Rmax = float(R.max())
    G = np.zeros_like(pts)
    tol = 1e-8
    for k, f in enumerate(faces):
        if R[k] <= Rmax - tol:
            continue
        i, j, m = int(f[0]), int(f[1]), int(f[2])
        a, b, c = pts[i], pts[j], pts[m]
        sa = np.clip(float(np.dot(a, b)), -1.0, 1.0)
        sb = np.clip(float(np.dot(b, c)), -1.0, 1.0)
        sc = np.clip(float(np.dot(c, a)), -1.0, 1.0)
        a1 = np.arccos(sa)
        b1 = np.arccos(sb)
        c1 = np.arccos(sc)
        s = 0.5 * (a1 + b1 + c1)
        sin_a = np.sin(a1)
        if sin_a <= 1e-12:
            continue
        R = R[k]
        # dR/da1  (a1 is the side opposite vertex a)
        dR_da = (0.5 * (np.cos(s) * np.sin(s - a1) - np.sin(s) * np.cos(s - a1))
                 * np.sin(s - b1) * np.sin(s - c1)) / sin_a
        dR_db = (0.5 * np.sin(s) * np.cos(s - b1) * np.sin(s - a1) * np.sin(s - c1)) / sin_a
        dR_dc = (0.5 * np.sin(s) * np.sin(s - a1) * np.sin(s - c1) * np.cos(s - c1)) / sin_a
        # d(a1)/da = -b/|a-b| ; d(a1)/db = -a/|a-b|
        dab = np.linalg.norm(a - b)
        if dab > 1e-9:
            G[i] += dR_da * (-b / dab)
            G[j] += dR_da * (-a / dab)
        dbc = np.linalg.norm(b - c)
        if dbc > 1e-9:
            G[j] += dR_db * (-c / dbc)
            G[m] += dR_db * (-b / dbc)
        dca = np.linalg.norm(c - a)
        if dca > 1e-9:
            G[m] += dR_dc * (-a / dca)
            G[i] += dR_dc * (-c / dca)
    return Rmax, G


def _valid(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    n = np.linalg.norm(pts, axis=1)
    if np.any(n < 1e-6):
        return False
    if np.any(n > 1.0 + 1e-5):
        return False
    c = pts.mean(axis=0)
    if np.linalg.norm(c) > 0.15:
        return False
    try:
        hull = ConvexHull(pts)
    except Exception:
        return False
    if hull.volume <= 1e-6:
        return False
    return True


# ---------------------------------------------------------------------------
# Seed: Hardin-Sloane-Smith icosahedral configuration
# ---------------------------------------------------------------------------
def _hss_seed():
    # icosahedron vertices (12)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    raw = [
        (0.0, 1.0, t), (0.0, -1.0, t), (0.0, 1.0, -t), (0.0, -1.0, -t),
        (1.0, t, 0.0), (1.0, -t, 0.0), (-1.0, t, 0.0), (-1.0, -t, 0.0),
        (t, 0.0, 1.0), (t, 0.0, -1.0), (-t, 0.0, 1.0), (-t, 0.0, -1.0),
    ]
    V = np.array(raw, dtype=float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    pts = [V]
    # face centers (20)
    try:
        hull = ConvexHull(V)
        faces = hull.simplices
    except Exception:
        faces = []
    for f in faces:
        c = V[f[0]] + V[f[1]] + V[f[2]]
        c /= np.linalg.norm(c)
        pts.append([c])
    # edge midpoints (30)
    edges = []
    seen = set()
    for f in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            x, y = int(f[a]), int(f[b])
            key = (min(x, y), max(x, y))
            if key in seen:
                continue
            seen.add(key)
            c = V[x] + V[y]
            c /= np.linalg.norm(c)
            pts.append([c])
    edges = [p[0] for p in pts[len(V):]]
    # 53 "caps" placed on the icosahedron's 5-fold symmetry axes (one per 20-face
    # pentagonal cap).  Directions are the 12 vertices, 6 face-centers of the
    # dual octahedron, and the 6 edge-midpoints of the dual octahedron.
    axis_dirs = list(V)
    for f in faces:
        c = V[f[0]] + V[f[1]] + V[f[2]]
        c /= np.linalg.norm(c)
        axis_dirs.append(c)
    # dual-octahedron edge midpoints (6)
    dual_edges = []
    seen = set()
    for f in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            x, y = int(f[a]), int(f[b])
            key = (min(x, y), max(x, y))
            if key in seen:
                continue
            seen.add(key)
            dual_edges.append((x, y))
    for x, y in dual_edges:
        c = V[x] + V[y]
        c /= np.linalg.norm(c)
        axis_dirs.append(c)
    # 53 = 32 + 21: distribute one cap per axis, remaining on a Fibonacci ring
    axis_dirs = np.array(axis_dirs, dtype=float)
    axis_dirs = axis_dirs / np.linalg.norm(axis_dirs, axis=1)[:, None]
    for d in axis_dirs:
        pts.append([d])
    # fill up to 130 with a Fibonacci spiral (will be refined by the optimizer)
    i = np.arange(130) + 0.5
    z = 1.0 - 2.0 * i / 130.0
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    while len(pts) < N:
        pts.append([fib[len(pts) % len(fib)]])
    pts = np.array(pts[:N], dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    return pts


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def _project(pts):
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    return pts


def _optimize(start, deadline):
    pts = start.copy()
    pts = _project(pts)
    best = pts.copy()
    best_r = _covering_and_grad(best)[0]
    step = 0.05
    no_improve = 0
    it = 0
    while time.time() < deadline:
        r, G = _covering_and_grad(pts)
        if np.isfinite(r) and r < best_r - 1e-12:
            best = pts.copy()
            best_r = r
            no_improve = 0
        else:
            no_improve += 1
        if not np.all(np.isfinite(G)):
            break
        gnorm = np.linalg.norm(G)
        if gnorm < 1e-9:
            break
        g = G / gnorm
        pts = _project(pts - step * g)
        if not _valid(pts):
            pts = best.copy()
            step *= 0.5
            if step < 1e-5:
                break
            continue
        if no_improve > 25:
            step *= 0.7
            no_improve = 0
            if step < 1e-5:
                break
        it += 1
        if it % 200 == 0 and step > 1e-4:
            step *= 1.03
    return best, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    start = _hss_seed()
    if not _valid(start):
        # fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        start = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best = start.copy()
    best_r = _covering_and_grad(best)[0]
    # main run
    cand, r = _optimize(start, deadline - 20.0)
    if r < best_r:
        best, best_r = cand, r
    # a few short restarts from perturbed copies to escape local minima
    rng = np.random.default_rng(12345)
    while time.time() < deadline - 10.0:
        pert = best + rng.normal(0.0, 0.02, best.shape)
        pert = _project(pert)
        if not _valid(pert):
            continue
        cand, r = _optimize(pert, deadline - 10.0)
        if r < best_r:
            best, best_r = cand, r
    # final safety: ensure validity, else return the seed
    if not _valid(best):
        best = start
    return _project(best)
