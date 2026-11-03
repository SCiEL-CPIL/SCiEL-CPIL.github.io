"""Packing 11 unit hexagons in a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin within 150s limit


def _hex_vertices_np(cx, cy, angle_deg=0.0):
    """Vertices of unit regular hexagon centered at (cx,cy) with rotation, as (6,2) array."""
    a = math.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_vertices(cx, cy, angle_deg=0.0):
    v = _hex_vertices_np(cx, cy, angle_deg)
    return [(v[i, 0], v[i, 1]) for i in range(6)]


def _all_vertices(inner):
    """Stack all 6N vertices into (6N, 2) array."""
    verts = []
    for x, y, a in inner:
        verts.append(_hex_vertices_np(x, y, a))
    return np.vstack(verts)


def _enclosing_side_and_center(inner, angle_deg):
    """For a given outer hex angle, find the minimum-side enclosing regular hexagon.
    Returns (cx, cy, side). Uses LP for exactness."""
    from scipy.optimize import linprog
    V = _all_vertices(inner)
    n = V.shape[0]
    c = np.array([0.0, 0.0, 1.0])  # minimize t (inradius)
    A_rows, b_rows = [], []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        proj = V[:, 0] * nx + V[:, 1] * ny
        A_rows.append(np.column_stack([-nx * np.ones(n), -ny * np.ones(n), -np.ones(n)]))
        b_rows.append(-proj)
        A_rows.append(np.column_stack([nx * np.ones(n), ny * np.ones(n), -np.ones(n)]))
        b_rows.append(proj)
    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_rows)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        cx, cy, t = res.x
        return cx, cy, t / (math.sqrt(3) / 2)
    # fallback: centroid
    cx, cy = V[:, 0].mean(), V[:, 1].mean()
    t = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        proj = (V[:, 0] - cx) * nx + (V[:, 1] - cy) * ny
        t = max(t, proj.max(), -proj.min())
    return cx, cy, t / (math.sqrt(3) / 2)


def _smooth_enclosing_side(inner, angle_deg, beta=20.0):
    """Smooth surrogate for enclosing side using log-sum-exp over face normals.
    This is differentiable and amenable to Nelder-Mead."""
    V = _all_vertices(inner)
    cx, cy = V[:, 0].mean(), V[:, 1].mean()
    # For each face normal direction, compute max and min projection
    # The required inradius is max over k of (max_proj_k - min_proj_k)/2
    # but centered optimally it's max over k of max(|max_proj|, |min_proj|) after centering
    # Actually for a regular hexagon, inradius = max over face normals of (max_proj - min_proj)/2
    # when centered at the midpoint. Let's use that.
    max_half_width = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        proj = (V[:, 0] - cx) * nx + (V[:, 1] - cy) * ny
        half_width = (proj.max() - proj.min()) / 2.0
        max_half_width = max(max_half_width, half_width)
    side = max_half_width / (math.sqrt(3) / 2)
    return side


def _min_outer_side(inner):
    """Find the minimum enclosing hexagon side over all angles.
    Uses smooth Nelder-Mead on the angle, then exact LP refinement."""
    # Coarse grid to find good starting angles
    best_side = float('inf')
    best_angle = 0.0
    for ang in np.arange(0.0, 60.0, 10.0):
        s = _smooth_enclosing_side(inner, float(ang), beta=20.0)
        if s < best_side:
            best_side = s
            best_angle = float(ang)
    # Nelder-Mead on the smooth objective
    def obj(x):
        return _smooth_enclosing_side(inner, x[0], beta=20.0)
    res = minimize(obj, [best_angle], method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000})
    if res.success:
        best_angle = float(res.x[0]) % 60.0
    # Exact LP refinement around the best angle
    final_best = float('inf')
    final_cx = final_cy = 0.0
    for da in np.arange(-2.0, 2.0, 0.1):
        ang = (best_angle + float(da)) % 60.0
        cx, cy, s = _enclosing_side_and_center(inner, ang)
        if s < final_best:
            final_best = s
            final_cx, final_cy, final_angle = cx, cy, ang
    # Even finer
    for da in np.arange(-0.2, 0.2, 0.02):
        ang = (final_angle + float(da)) % 60.0
        cx, cy, s = _enclosing_side_and_center(inner, ang)
        if s < final_best:
            final_best = s
            final_cx, final_cy, final_angle = cx, cy, ang
    return final_cx, final_cy, final_best, final_angle


def _disjoint(inner, eps=1e-9):
    """Check all inner hexagons have disjoint interiors."""
    polys = [_hex_vertices(x, y, a) for x, y, a in inner]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(polys[i], polys[j], eps):
                return False
    return True


def _overlap(p1, p2, eps=1e-9):
    for v in p1:
        if _in_convex(v, p2, eps):
            return True
    for v in p2:
        if _in_convex(v, p1, eps):
            return True
    for i in range(6):
        for j in range(6):
            if _seg_intersect(p1[i], p1[(i + 1) % 6], p2[j], p2[(j + 1) % 6], eps):
                return True
    return False


def _in_convex(pt, poly, eps=1e-9):
    px, py = pt
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True


def _seg_intersect(p1, p2, p3, p4, eps=1e-9):
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cr(p3, p4, p1), cr(p3, p4, p2)
    d3, d4 = cr(p1, p2, p3), cr(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_seg(p3, p1, p4, eps):
        return True
    if abs(d2) <= eps and _on_seg(p3, p2, p4, eps):
        return True
    if abs(d3) <= eps and _on_seg(p1, p3, p2, eps):
        return True
    if abs(d4) <= eps and _on_seg(p1, p4, p2, eps):
        return True
    return False


def _on_seg(p, q, r, eps=1e-9):
    return (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
            min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
            abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) <= eps)


def _build_hex_cluster(r2_indices):
    """Build 11 hexagons: center + 6 ring-1 + 4 chosen ring-2 positions.
    r2_indices: 4 indices into the 12 ring-2 positions (6 corners + 6 edges)."""
    s3 = math.sqrt(3)
    inner = [[0.0, 0.0, 0.0]]
    # Ring 1: 6 hexagons at distance s3, angles 0,60,...,300
    for k in range(6):
        a = math.radians(60 * k)
        inner.append([s3 * math.cos(a), s3 * math.sin(a), 0.0])
    # Ring 2: 12 positions. Corners at distance 2*s3, angles 0,60,...,300
    # Edges at distance s3*sqrt(3), angles 30,90,...,330
    ring2 = []
    for k in range(6):  # corners
        a = math.radians(60 * k)
        ring2.append((2 * s3 * math.cos(a), 2 * s3 * math.sin(a)))
    for k in range(6):  # edges
        a = math.radians(60 * k + 30)
        ring2.append((s3 * math.sqrt(3) * math.cos(a), s3 * math.sqrt(3) * math.sin(a)))
    for idx in r2_indices:
        x, y = ring2[idx]
        inner.append([x, y, 0.0])
    return inner


def _local_polish(inner, outer_angle, max_iter=500, tol=1e-10):
    """Coordinate descent on positions to reduce enclosing side."""
    best = [h[:] for h in inner]
    _, _, best_side, _ = _min_outer_side(best)
    step = 0.05
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        improved = False
        for i in range(N):
            for p in range(3):
                for d in (-1, 1):
                    trial = [h[:] for h in best]
                    trial[i][p] += d * step
                    if _disjoint(trial):
                        _, _, s, _ = _min_outer_side(trial)
                        if s < best_side - 1e-11:
                            best = trial
                            best_side = s
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return best, best_side


def construct():
    global _START_TIME, _DEADLINE
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Enumerate all C(12,4) = 495 choices of 4 ring-2 positions
    from itertools import combinations
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    candidates = list(combinations(range(12), 4))
    # Shuffle to spread out work
    import random
    random.seed(42)
    random.shuffle(candidates)

    for combo in candidates:
        if time.time() > _DEADLINE - 20.0:
            break
        inner = _build_hex_cluster(combo)
        if not _disjoint(inner):
            continue
        cx, cy, s, ang = _min_outer_side(inner)
        if s < best_side:
            best_side = s
            best_inner = [h[:] for h in inner]
            best_center = (cx, cy)
            best_angle = ang

    if best_inner is None:
        # fallback: default cluster
        best_inner = _build_hex_cluster([0, 2, 4, 6])
        cx, cy, s, ang = _min_outer_side(best_inner)
        best_side = s
        best_center = (cx, cy)
        best_angle = ang

    # Local polish
    if time.time() < _DEADLINE - 25.0:
        polished, ps = _local_polish(best_inner, best_angle, max_iter=300)
        if ps < best_side - 1e-9:
            best_inner = polished
            best_side = ps
            cx, cy, s, ang = _min_outer_side(best_inner)
            best_center = (cx, cy)
            best_angle = ang

    # Final exact LP for the best configuration
    if time.time() < _DEADLINE - 5.0:
        cx, cy, s, ang = _min_outer_side(best_inner)
        if s < best_side:
            best_side = s
            best_center = (cx, cy)
            best_angle = ang

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
