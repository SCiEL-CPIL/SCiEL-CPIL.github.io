"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_TIME_LIMIT = 118.0  # safety margin below the 150s wall-clock kill


def _normalize(points):
    n = np.linalg.norm(points, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return points / n


def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _facet_circumdata(hull):
    """Return (circumcenters, circumradii, facet_points) for each convex hull facet.

    A facet is invalid (empty-cap / hemisphere issue) if its circumradius is >= pi/2
    or the circumcenter is not on the outward side (normal . circumcenter < 0).
    """
    pts = hull.points
    simp = hull.simplices
    centers = np.empty((len(simp), 3))
    radii = np.empty(len(simp))
    for k, tri in enumerate(simp):
        a = pts[tri[0]]
        b = pts[tri[1]]
        c = pts[tri[2]]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            centers[k] = (a + b + c) / 3.0
            radii[k] = np.pi
            continue
        nrm = nrm / nn
        # outward normal
        centroid = (a + b + c) / 3.0
        if np.dot(nrm, centroid) < 0:
            nrm = -nrm
        # solve circumcenter on the plane: x = a + u*(b-a) + v*(c-a), equidistant
        # |x-a|^2 = |x-b|^2 -> 2 x.(b-a) = |b-a|^2 + |a|^2 - |b|^2
        # Since all on unit sphere, |a|^2=|b|^2=|c|^2=1 -> |b-a|^2 = 2 - 2 a.b
        d1 = b - a
        d2 = c - a
        # equations: x.d1 = d1.(a+d1/2) ; x.d2 = d2.(a+d2/2)
        rhs1 = np.dot(d1, a + 0.5 * d1)
        rhs2 = np.dot(d2, a + 0.5 * d2)
        A = np.array([[np.dot(d1, d1), np.dot(d1, d2)],
                      [np.dot(d2, d1), np.dot(d2, d2)]])
        try:
            uv = np.linalg.solve(A, np.array([rhs1, rhs2]))
        except np.linalg.LinAlgError:
            centers[k] = centroid
            radii[k] = np.pi
            continue
        x = a + uv[0] * d1 + uv[1] * d2
        x = x / (np.linalg.norm(x) + 1e-15)
        centers[k] = x
        # circumradius = angle from a to x
        cosv = np.clip(np.dot(x, a), -1.0, 1.0)
        radii[k] = np.arccos(cosv)
    return centers, radii, simp


def _covering_radius(points):
    """Return (covering_radius, per-facet radii, per-facet point indices, centers)."""
    pts = _normalize(points)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None, None, None
    centers, radii, simp = _facet_circumdata(hull)
    # mark invalid facets (radius >= pi/2 or center wrong side)
    for k in range(len(simp)):
        tri = simp[k]
        centroid = pts[tri].mean(axis=0)
        if np.dot(centers[k], centroid) < 0 or radii[k] >= np.pi / 2.0:
            radii[k] = np.pi
    return radii.max(), radii, simp, centers


def _softmax_radius(radii, beta):
    # smooth max
    m = radii.max()
    return m + np.log(np.exp((radii - m) * beta).sum()) / beta


def _objective(x, beta, div):
    pts = _normalize(x)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1e6
    centers, radii, simp = _facet_circumdata(hull)
    for k in range(len(simp)):
        tri = simp[k]
        centroid = pts[tri].mean(axis=0)
        if np.dot(centers[k], centroid) < 0 or radii[k] >= np.pi / 2.0:
            radii[k] = np.pi
    cr = _softmax_radius(radii, beta)
    # diversity penalty: penalize small pairwise separations
    # use a cheap proxy: penalize min pair distance
    d = np.einsum('ij,ij->i', pts, pts)
    # avoid full NxN cost blowup; N=50 fine
    P = pts @ pts.T
    np.fill_diagonal(P, 1.0)
    mind = np.sqrt(np.maximum(0.0, 2.0 - P.min()))
    div_pen = div * np.exp(-mind / 0.05)
    return float(cr + div_pen)


def _lloyd_step(points, iters=200, decay=True):
    pts = _normalize(points.copy())
    best = pts.copy()
    best_cr, _, _, _ = _covering_radius(pts)
    for it in range(iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        centers, radii, simp = _facet_circumdata(hull)
        # for each point, find worst incident facet
        # build point -> facet map
        point_worst = np.full(N, -1.0)
        for k in range(len(simp)):
            r = radii[k]
            if r < np.pi / 2.0:
                for p in simp[k]:
                    if r > point_worst[p]:
                        point_worst[p] = r
        # move each point toward its worst facet circumcenter
        moved = pts.copy()
        alpha = 0.35 if not decay else max(0.02, 0.35 * (1.0 - it / float(iters)))
        for p in range(N):
            if point_worst[p] < 0:
                continue
            # find the facet
            # locate center of that radius
            # re-find: pick center whose radius matches and contains p
            tgt = None
            for k in range(len(simp)):
                if p in simp[k] and abs(radii[k] - point_worst[p]) < 1e-12:
                    tgt = centers[k]
                    break
            if tgt is None:
                continue
            v = pts[p] + (tgt - pts[p]) * alpha
            v = v / (np.linalg.norm(v) + 1e-15)
            moved[p] = v
        # evaluate
        cr, _, _, _ = _covering_radius(moved)
        if cr < best_cr - 1e-10:
            best = moved.copy()
            best_cr = cr
            pts = moved
        else:
            # try smaller step
            moved2 = pts.copy()
            for p in range(N):
                if point_worst[p] < 0:
                    continue
                tgt = None
                for k in range(len(simp)):
                    if p in simp[k] and abs(radii[k] - point_worst[p]) < 1e-12:
                        tgt = centers[k]
                        break
                if tgt is None:
                    continue
                v = pts[p] + (tgt - pts[p]) * (alpha * 0.3)
                v = v / (np.linalg.norm(v) + 1e-15)
                moved2[p] = v
            cr2, _, _, _ = _covering_radius(moved2)
            if cr2 < best_cr - 1e-10:
                best = moved2.copy()
                best_cr = cr2
                pts = moved2
            else:
                break
    return best, best_cr


def _local_opt(points, beta, div, t0):
    x0 = _normalize(points).ravel()
    def obj(x):
        if time.time() - t0 > _TIME_LIMIT - 5:
            raise TimeoutError
        return _objective(x, beta, div)
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 400, 'maxfun': 8000, 'gtol': 1e-8})
        if res.x is not None:
            cand = _normalize(res.x.reshape(-1, 3))
            cr, _, _, _ = _covering_radius(cand)
            return cand, cr
    except Exception:
        pass
    return points, _covering_radius(points)[0]


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    starts = [_fibonacci_points(N)]
    for _ in range(6):
        starts.append(_normalize(rng.normal(size=(N, 3))))

    best_pts = None
    best_cr = np.pi
    for s in starts:
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
        # phase 1: Lloyd relaxation
        lp, lcr = _lloyd_step(s, iters=120)
        if lcr < best_cr:
            best_cr = lcr
            best_pts = lp
        # phase 2: local optimization with a few beta values
        for beta in (12.0, 25.0):
            if time.time() - t0 > _TIME_LIMIT - 15:
                break
            op, ocr = _local_opt(lp, beta, div=0.0)
            if ocr < best_cr:
                best_cr = ocr
                best_pts = op
        # phase 3: refine best with a final Lloyd + light opt
        if time.time() - t0 > _TIME_LIMIT - 12:
            break
        rp, rcr = _lloyd_step(best_pts, iters=60)
        if rcr < best_cr:
            best_cr = rcr
            best_pts = rp

    # final safety refinement on the global best
    if time.time() - t0 < _TIME_LIMIT - 8 and best_pts is not None:
        fp, fcr = _local_opt(best_pts, 25.0, div=0.0)
        if fcr < best_cr:
            best_cr = fcr
            best_pts = fp

    if best_pts is None:
        best_pts = _fibonacci_points(N)
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
