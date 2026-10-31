import math
import time
import random
import numpy as np

N = 11
DEADLINE = 140.0  # safety margin below the 150s kill


def _verts(px, py, s, a):
    """Vertices of a regular hexagon, center (px,py), side s, rotation a (rad)."""
    return np.array([[px + s * math.cos(a + i * math.pi / 3.0),
                      py + s * math.sin(a + i * math.pi / 3.0)] for i in range(6)])


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _cross(p3, p1, p4) >= 0 and _cross(p3, p1, p4) <= 0 and \
       (p1[0] <= max(p3[0], p4[0]) and p1[0] >= min(p3[0], p4[0]) and
        p1[1] <= max(p3[1], p4[1]) and p1[1] >= min(p3[1], p4[1])):
        return True
    if d2 == 0 and _cross(p3, p2, p4) >= 0 and _cross(p3, p2, p4) <= 0 and \
       (p2[0] <= max(p3[0], p4[0]) and p2[0] >= min(p3[0], p4[0]) and
        p2[1] <= max(p3[1], p4[1]) and p2[1] >= min(p3[1], p4[1])):
        return True
    if d3 == 0 and _cross(p1, p3, p2) >= 0 and _cross(p1, p3, p2) <= 0 and \
       (p3[0] <= max(p1[0], p2[0]) and p3[0] >= min(p1[0], p2[0]) and
        p3[1] <= max(p1[1], p2[1]) and p3[1] >= min(p1[1], p2[1])):
        return True
    if d4 == 0 and _cross(p1, p4, p2) >= 0 and _cross(p1, p4, p2) <= 0 and \
       (p4[0] <= max(p1[0], p2[0]) and p4[0] >= min(p1[0], p2[0]) and
        p4[1] <= max(p1[1], p2[1]) and p4[1] >= min(p1[1], p2[1])):
        return True
    return False


def _polygons_overlap(a, b):
    n = len(a)
    m = len(b)
    for i in range(n):
        for j in range(m):
            if _segments_intersect(a[i], a[(i + 1) % n], b[j], b[(j + 1) % m]):
                return True
    # one strictly inside the other
    if _cross(b[0], b[1], b[2]) > 0:
        if all(_cross(b[0], b[1], a[0]) >= 0 and _cross(b[1], b[2], a[0]) >= 0 and
               _cross(b[2], b[0], a[0]) >= 0):
            return True
    else:
        if all(_cross(b[0], b[1], a[0]) <= 0 and _cross(b[1], b[2], a[0]) <= 0 and
               _cross(b[2], b[0], a[0]) <= 0):
            return True
    if _cross(a[0], a[1], a[2]) > 0:
        if all(_cross(a[0], a[1], b[0]) >= 0 and _cross(a[1], a[2], b[0]) >= 0 and
               _cross(a[2], a[0], b[0]) >= 0):
            return True
    else:
        if all(_cross(a[0], a[1], b[0]) <= 0 and _cross(a[1], a[2], b[0]) <= 0 and
               _cross(a[2], a[0], b[0]) <= 0):
            return True
    return False


def _hex_inside_hex(inner, outer):
    """True if regular hexagon 'inner' is fully inside regular hexagon 'outer'."""
    for v in inner:
        if not _polygons_overlap(_verts(v[0], v[1], 0.5, 0.0), outer):
            return False
    return True


def _outer_side_for_cluster(verts, theta):
    """Smallest outer side S (centered at origin) with rotation theta containing all cluster vertices."""
    # outward unit normals of the outer hexagon
    nrm = np.array([[math.cos(theta + i * math.pi / 3.0),
                     math.sin(theta + i * math.pi / 3.0)] for i in range(6)])
    # support function of the cluster in each normal direction
    h = np.array([np.max(verts @ n) for n in nrm])
    # outer vertex along normal i is at S * n_i ; require S * n_i . n_i >= h_i
    # n_i . n_i = 1  =>  S >= h_i
    return float(np.max(h))


def _build_cluster():
    """Central hexagon + ring of 10 touching it. Returns list of (x,y,angle_deg)."""
    r = 2.0  # center-to-center distance for two touching unit hexagons
    pts = [(0.0, 0.0)]
    for i in range(10):
        ang = i * math.pi / 5.0
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    return [[x, y, 0.0] for (x, y) in pts]


def _cluster_valid():
    """Verify the 11 cluster hexagons are pairwise disjoint (touching edges allowed)."""
    base = _build_cluster()
    verts = [_verts(x, y, 1.0, 0.0) for (x, y, _) in base]
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if _polygons_overlap(verts[i], verts[j]):
                return False
    return True


def _evaluate(cluster_pts, theta):
    """Return outer side S for a cluster (list of (x,y)) rotated by theta, centered at origin."""
    c = np.array(cluster_pts, dtype=float)
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    verts = (c @ R.T)
    return _outer_side_for_cluster(verts, theta)


def _optimize_theta(cluster_pts, t0):
    """Minimize outer side over theta via golden-section search over [0, 2pi]."""
    lo, hi = 0.0, 2.0 * math.pi
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo + gr * (hi - lo), hi - gr * (hi - lo)
    fa, fb = _evaluate(cluster_pts, a), _evaluate(cluster_pts, b)
    while (b - a) > 1e-4 and (time.time() - t0) < DEADLINE:
        if fa < fb:
            hi = b
            b = a
            fb = fa
            a = lo + gr * (hi - lo)
            fa = _evaluate(cluster_pts, a)
        else:
            lo = a
            a = b
            fa = fb
            b = hi - gr * (hi - lo)
            fb = _evaluate(cluster_pts, b)
    theta = 0.5 * (a + b)
    return _evaluate(cluster_pts, theta), theta


def _jitter_cluster(cluster_pts, t0):
    """Local search: perturb cluster, keep best outer side found."""
    best_pts = [list(p) for p in cluster_pts]
    best_s = _evaluate(best_pts, 0.0)[0]
    step = 0.15
    rng = random.Random(12345)
    iters = 0
    while (time.time() - t0) < DEADLINE and iters < 6000:
        iters += 1
        cand = [list(p) for p in best_pts]
        for k in range(3):
            idx = rng.randrange(len(cand))
            cand[idx][0] += (rng.random() - 0.5) * 2 * step
            cand[idx][1] += (rng.random() - 0.5) * 2 * step
        # quick overlap check using vertex support approx is expensive; do full check
        verts = [_verts(x, y, 1.0, 0.0) for (x, y) in cand]
        ok = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if _polygons_overlap(verts[i], verts[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            step *= 0.97
            continue
        s, _ = _optimize_theta(cand, t0)
        if s < best_s:
            best_s = s
            best_pts = cand
            step = min(step * 1.05, 0.25)
        else:
            step *= 0.97
    return best_s, best_pts


def construct():
    t0 = time.time()
    # verify the base cluster is a valid (disjoint) packing
    assert _cluster_valid(), "base cluster overlaps"
    base = _build_cluster()

    # 1) global rotation search on the symmetric cluster
    best_s, best_theta = _optimize_theta(base, t0)

    # 2) local search on cluster positions (rigid-rotation still applied)
    if (time.time() - t0) < DEADLINE:
        s2, pts2 = _jitter_cluster(base, t0)
        if s2 < best_s:
            best_s = s2
            # recompute theta for the jittered cluster
            best_s, best_theta = _optimize_theta(pts2, t0)

    inner = [[pts2[0], pts2[1], 0.0], [pts2[1], pts2[2], 0.0]] if False else \
            [[x, y, 0.0] for (x, y) in pts2]

    # outer hexagon centered at origin, side best_s, rotation best_theta
    return inner, (0.0, 0.0), best_s, math.degrees(best_theta)


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
