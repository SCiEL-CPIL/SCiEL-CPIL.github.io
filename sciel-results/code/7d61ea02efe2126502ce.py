"""Covering the unit sphere with 50 equal caps - Riemannian worst-facet flow."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 50
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_degrees(pts):
    """Covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        # each facet: list of point indices
        best = 0.0
        for f in hull.simplices:
            # f is a 3-simplex (4 points) in 3D
            a, b, c, d = pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]
            # circumcenter of the 4 points (sphere through them)
            # solve for center o minimizing |o - p|^2 equal
            # |o-a|^2 = |o-b|^2 => 2 o.(b-a) = |b|^2 - |a|^2
            M = np.array([b - a, c - a, d - a])
            rhs = 0.5 * (np.sum(b * b) - np.sum(a * a),
                         np.sum(c * c) - np.sum(a * a),
                         np.sum(d * d) - np.sum(a * a))
            try:
                o = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            r = np.linalg.norm(o - a)
            # angular radius: half-angle subtended by the cap
            # cap center is o/|o|, angular radius = angle between o and a
            on = np.linalg.norm(o)
            if on < 1e-12:
                continue
            cosang = np.clip(np.dot(o / on, a), -1.0, 1.0)
            ang = np.arccos(cosang)  # radians
            deg = np.degrees(ang)
            if deg > best:
                best = deg
        return best
    except Exception:
        return np.inf


def _worst_facet(pts):
    """Return (facet_indices, circumcenter_unit, cap_radius_deg) for the largest empty cap."""
    try:
        hull = ConvexHull(pts)
        best_deg = -1.0
        best_f = None
        best_c = None
        for f in hull.simplices:
            a, b, c, d = pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]
            M = np.array([b - a, c - a, d - a])
            rhs = 0.5 * (np.sum(b * b) - np.sum(a * a),
                         np.sum(c * c) - np.sum(a * a),
                         np.sum(d * d) - np.sum(a * a))
            try:
                o = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            on = np.linalg.norm(o)
            if on < 1e-12:
                continue
            cosang = np.clip(np.dot(o / on, a), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            if ang > best_deg:
                best_deg = ang
                best_f = f
                best_c = o / on
        return best_f, best_c, best_deg
    except Exception:
        return None, None, -1.0


def _tammes_push(pts, step):
    """Max-min (Tammes) gradient: each point moves away from its nearest neighbor."""
    p = pts.copy()
    n = len(p)
    # pairwise distances
    d = 1.0 - np.clip(p @ p.T, -1.0, 1.0)
    np.fill_diagonal(d, np.inf)
    # for each point, find nearest neighbor and push away
    for i in range(n):
        j = int(np.argmin(d[i]))
        # direction: from nearest neighbor j toward i, projected tangent
        v = p[i] - p[j]
        # tangent component (remove radial)
        v = v - np.dot(v, p[i]) * p[i]
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            # random tangent
            t = np.random.randn(3)
            t = t - np.dot(t, p[i]) * p[i]
            t = t / (np.linalg.norm(t) + 1e-12)
        else:
            t = v / vn
        p[i] = p[i] + step * t
    return _normalize(p)


def _reduce_worst_cap(pts, step):
    """Move the worst facet's vertices toward the cap circumcenter."""
    f, c, deg = _worst_facet(pts)
    if f is None or c is None:
        return pts
    p = pts.copy()
    for idx in f:
        # direction from vertex toward cap center, tangent
        v = c - p[idx]
        v = v - np.dot(v, p[idx]) * p[idx]
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        p[idx] = p[idx] + step * (v / vn)
    return _normalize(p)


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    best_pts = _fibonacci_start()
    best_cr = _covering_radius_degrees(best_pts)

    # A couple of diverse starts, keep the best
    starts = [best_pts]
    for _ in range(2):
        cand = _normalize(rng.normal(size=(N, 3)))
        cr = _covering_radius_degrees(cand)
        if cr < best_cr:
            best_cr = cr
            best_pts = cand
        starts.append(cand)

    # Main loop: alternate worst-cap reduction and Tammes push with annealing step
    step = 0.05
    it = 0
    while time.time() - t0 < _DEADLINE - 5.0:
        it += 1
        # Reduce worst cap
        cand = _reduce_worst_cap(best_pts, step)
        cr = _covering_radius_degrees(cand)
        if cr < best_cr - 1e-10:
            best_cr = cr
            best_pts = cand
        # Tammes push to spread out
        cand2 = _tammes_push(best_pts, step * 0.5)
        cr2 = _covering_radius_degrees(cand2)
        if cr2 < best_cr - 1e-10:
            best_cr = cr2
            best_pts = cand2
        # anneal step size
        step = max(0.002, step * 0.995)
        # occasional re-jump if stuck
        if it % 200 == 0:
            cand3 = _reduce_worst_cap(_tammes_push(best_pts, 0.02), 0.02)
            cr3 = _covering_radius_degrees(cand3)
            if cr3 < best_cr - 1e-10:
                best_cr = cr3
                best_pts = cand3

    return _normalize(best_pts)
