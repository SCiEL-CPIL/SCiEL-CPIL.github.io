"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.
Combines SAT-based collision with SDF-smoothed L-BFGS-B optimization and diverse seeds.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

N = 12
DEADLINE = 115.0  # safety margin before the 120s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices_local(theta):
    """Vertices of unit regular hexagon (flat-top) at origin, rotated by theta."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    """6 outward edge-normal directions of a regular hexagon oriented by theta."""
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side(pts, theta, center):
    """Side length of smallest regular hexagon (orientation theta, center) containing pts."""
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = center @ nrm.T
    req = (P - c).max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _hexagon_overlap(A, B):
    """SAT test: True if two convex hexagons (arrays of (x,y)) overlap."""
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() - 1e-9 or projB.max() < projA.min() - 1e-9:
                return False
    return True

def _min_translation(A, B):
    """Separating axis + signed penetration depth (negative if overlapping)."""
    best_pen = -1.0
    best_axis = None
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            nl = math.hypot(n[0], n[1])
            if nl < 1e-9:
                continue
            n = n / nl
            pa = A @ n
            pb = B @ n
            sep = min(pa.max() - pb.min(), pb.max() - pa.min())
            if sep < 0 and sep > best_pen:
                best_pen = sep
                best_axis = n
    return best_pen, best_axis

# ---------------------------------------------------------------------------
# SDF for smooth optimization
# ---------------------------------------------------------------------------
def _sdf_hex_point(px, py, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at origin with rotation angle_deg.
    Negative = inside, positive = outside."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Transform point into hexagon's local frame
    lx = ca * px + sa * py
    ly = -sa * px + ca * py
    apothem = side * math.sqrt(3) / 2.0

    # Check if inside: for all 6 edges, projection <= apothem
    inside = True
    max_proj = 0.0
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx, ny = math.cos(theta), math.sin(theta)
        proj = nx * lx + ny * ly
        if proj > apothem + 1e-12:
            inside = False
        if proj > max_proj:
            max_proj = proj

    if inside:
        return -(apothem - max_proj)
    else:
        # Outside: compute distance to nearest edge segment
        min_d = float('inf')
        for k in range(6):
            t1 = math.radians(60 * k)
            t2 = math.radians(60 * (k + 1))
            v1 = np.array([side * math.cos(t1), side * math.sin(t1)])
            v2 = np.array([side * math.cos(t2), side * math.sin(t2)])
            edge = v2 - v1
            len2 = float(edge @ edge)
            if len2 < 1e-12:
                continue
            t = max(0.0, min(1.0, float(np.array([lx, ly]) @ edge / len2 - v1 @ edge / len2)))
            # Actually: t = dot(P - V1, edge) / dot(edge, edge)
            t = max(0.0, min(1.0, float(np.array([lx - v1[0], ly - v1[1]]) @ edge / len2)))
            q = v1 + t * edge
            d = math.hypot(lx - q[0], ly - q[1])
            if d < min_d:
                min_d = d
        return min_d

def _sdf_hex_hex(cx1, cy1, a1_deg, cx2, cy2, a2_deg, side=1.0):
    """Approximate SDF between two hexagons using vertex sampling."""
    verts1 = _hex_vertices_local(math.radians(a1_deg)) + np.array([cx1, cy1])
    verts2 = _hex_vertices_local(math.radians(a2_deg)) + np.array([cx2, cy2])
    min_d = float('inf')
    for v in verts1:
        d = _sdf_hex_point(v[0] - cx2, v[1] - cy2, a2_deg, side)
        if d < min_d:
            min_d = d
    for v in verts2:
        d = _sdf_hex_point(v[0] - cx1, v[1] - cy1, a1_deg, side)
        if d < min_d:
            min_d = d
    return min_d

# ---------------------------------------------------------------------------
# Seed configurations
# ---------------------------------------------------------------------------
def _seed(pattern):
    d = 2.0 * math.sqrt(3.0) + 1e-6
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    if pattern == "rect":
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
    elif pattern == "hex":
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
    elif pattern == "rect_tall":
        for i in range(-1, 2):
            for j in range(-2, 2):
                pts.append(i * a1 + j * a2)
    elif pattern == "ring":
        # 12 hexagons arranged in a ring around center
        for k in range(12):
            ang = 2.0 * math.pi * k / 12.0
            r = 2.5 * math.sqrt(3.0)
            pts.append(np.array([r * math.cos(ang), r * math.sin(ang)]))
    elif pattern == "strip":
        # 2 rows of 6
        for i in range(6):
            pts.append(np.array([i * d, 0.0]))
            pts.append(np.array([i * d, d * math.sqrt(3.0)]))
    else:
        # "rotated_grid" - triangular lattice rotated 30 degrees
        a1r = np.array([d * math.cos(math.pi/6), d * math.sin(math.pi/6)])
        a2r = np.array([-d * math.sin(math.pi/6), d * math.cos(math.pi/6)])
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1r + j * a2r)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return [[float(pts[k][0]), float(pts[k][1]), 0.0] for k in range(N)]

# ---------------------------------------------------------------------------
# Physical settling (SAT-based)
# ---------------------------------------------------------------------------
def _simulate(inner, outer, side, outer_theta, dt=0.02, steps=3000):
    centers = np.array([[r[0], r[1]] for r in inner], dtype=float)
    angles = np.array([math.radians(r[2]) for r in inner])
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    outer_pts = np.array(outer, dtype=float) * side
    verts = [_hex_vertices_local(angles[k]) for k in range(N)]

    for s in range(steps):
        if time.time() > DEADLINE:
            break
        # forces
        for i in range(N):
            vel[i] += -0.02 * (centers[i] - outer_pts)
            omega[i] += -0.005 * angles[i]
        # pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i] + centers[i]
                B = verts[j] + centers[j]
                if _hexagon_overlap(A, B):
                    pen, axis = _min_translation(A, B)
                    if axis is not None:
                        f = 0.5 * (-pen)
                        vel[i] += f * axis
                        vel[j] -= f * axis
        # containment
        for i in range(N):
            pts_i = verts[i] + centers[i]
            nrm = _norms(outer_theta)
            P = pts_i @ nrm.T
            c = outer_pts @ nrm.T
            for k in range(6):
                exceed = (P[:, k] - c[k]).max() - side * math.sqrt(3.0) / 2.0
                if exceed > 0:
                    n = nrm[k]
                    vel[i] -= 0.5 * exceed * n
        # integrate
        vel *= 0.85
        omega *= 0.85
        centers += vel * dt
        angles += omega * dt
        for k in range(N):
            verts[k] = _hex_vertices_local(angles[k])

    return [[float(centers[k][0]), float(centers[k][1]), float(math.degrees(angles[k]))] for k in range(N)]

# ---------------------------------------------------------------------------
# Smooth SDF-based objective for L-BFGS-B
# ---------------------------------------------------------------------------
def _smooth_objective(v, sigma=0.05):
    """
    v = [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, outer_angle]
    Objective = outer_side + penalty * sum(exp(-d/sigma) - 1) for overlapping pairs
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    cx, cy = v[3*N], v[3*N+1]
    outer_angle = v[3*N+2]

    # Compute outer side length
    all_pts = []
    for (px, py, pa) in cfg:
        verts = _hex_vertices_local(math.radians(pa)) + np.array([px, py])
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    s = _bounding_side(all_pts, outer_angle, np.array([cx, cy]))

    # Collision penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sdf_hex_hex(cfg[i][0], cfg[i][1], cfg[i][2],
                            cfg[j][0], cfg[j][1], cfg[j][2])
            if d < 3.0 * sigma:
                penalty += math.exp(-d / sigma) - 1.0

    return s + 10.0 * penalty

# ---------------------------------------------------------------------------
# L-BFGS-B optimization with smooth objective
# ---------------------------------------------------------------------------
def _refine_sdf(inner, outer, side, outer_theta, maxiter=500):
    """Use L-BFGS-B on the smooth SDF objective."""
    verts = [_hex_vertices_local(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])
    angles = np.array([math.radians(r[2]) for r in inner])

    # Initial vector: centers, angles (rad), outer center, outer angle
    v0 = np.zeros(3 * N + 3)
    for k in range(N):
        v0[3*k] = centers[k][0]
        v0[3*k+1] = centers[k][1]
        v0[3*k+2] = angles[k]
    v0[3*N] = outer[0]
    v0[3*N+1] = outer[1]
    v0[3*N+2] = outer_theta

    # Bounds: keep things in reasonable range
    bounds = [(-10, 10)] * (3 * N) + [(-5, 5), (-5, 5), (0, math.pi/3)]

    res = _scipy_minimize(_smooth_objective, v0, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})

    # Extract result
    new_inner = []
    for k in range(N):
        new_inner.append([float(res.x[3*k]), float(res.x[3*k+1]), float(math.degrees(res.x[3*k+2]))])
    new_center = (float(res.x[3*N]), float(res.x[3*N+1]))
    new_angle = float(res.x[3*N+2])

    # Compute actual side
    pts, _ = _all_points(new_inner)
    s = _bounding_side(pts, new_angle, np.array(new_center))

    return new_inner, new_center, s, new_angle

# ---------------------------------------------------------------------------
# Nelder-Mead refinement on centers (outer fixed)
# ---------------------------------------------------------------------------
def _all_points(inner):
    verts = [_hex_vertices_local(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])
    pts = []
    for k in range(N):
        pts.append(verts[k] + centers[k])
    return np.vstack(pts), centers

def _refine_inner(inner, outer, side, outer_theta, maxiter=200):
    """Nelder-Mead on the N centers, minimizing the bounding side (outer fixed)."""
    verts = [_hex_vertices_local(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])

    def obj(x):
        c = x.reshape(N, 2)
        pts = []
        for k in range(N):
            pts.append(verts[k] + c[k])
        pts = np.vstack(pts)
        return _bounding_side(pts, outer_theta, np.array(outer))

    x0 = centers.reshape(-1)
    res = _scipy_minimize(obj, x0, method="Nelder-Mead",
                          options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-9})
    c = res.x.reshape(N, 2)
    new_inner = []
    for k in range(N):
        new_inner.append([float(c[k][0]), float(c[k][1]), inner[k][2]])
    return new_inner

def _refine_outer(inner, outer, side, outer_theta):
    """Sweep outer angle, return best (side, center, angle)."""
    pts, centers = _all_points(inner)
    best = (side, outer, outer_theta)
    for ang in np.linspace(0.0, math.pi / 3.0, 121):
        b = _bounding_side(pts, ang, np.array(outer))
        if b < best[0] - 1e-9:
            best = (b, tuple(float(x) for x in outer), float(ang))
    return best

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(inner, outer, side, outer_theta):
        nonlocal best_side, best_inner, best_center, best_angle
        if side < best_side:
            best_side = side
            best_inner = [list(r) for r in inner]
            best_center = tuple(float(x) for x in outer)
            best_angle = float(outer_theta)

    patterns = ["rect", "hex", "rect_tall", "ring", "strip", "rotated_grid"]
    for pat in patterns:
        if time.time() > DEADLINE:
            break

        inner = _seed(pat)

        # Initial outer estimate via coarse angle sweep at origin
        pts, _ = _all_points(inner)
        s0, t0 = 10.0, 0.0
        for ang in np.linspace(0.0, math.pi / 3.0, 61):
            b = _bounding_side(pts, ang, np.zeros(2))
            if b < s0:
                s0, t0 = b, ang
        outer = (0.0, 0.0)

        # Phase 1: Physical settling
        inner = _simulate(inner, outer, s0, t0, steps=2000)

        # Phase 2: SDF-smoothed L-BFGS-B optimization
        if time.time() < DEADLINE - 10:
            inner, center1, s1, t1 = _refine_sdf(inner, outer, s0, t0, maxiter=300)
            _consider(inner, center1, s1, t1)

        # Phase 3: Nelder-Mead refinement on centers
        s1, outer1, t1 = _refine_outer(inner, center1, s1, t1)
        inner = _refine_inner(inner, outer1, s1, t1, maxiter=150)
        s2, outer2, t2 = _refine_outer(inner, outer1, s1, t1)
        inner = _refine_inner(inner, outer2, s2, t2, maxiter=100)
        s3, outer3, t3 = _refine_outer(inner, outer2, s2, t2)

        _consider(inner, outer3, s3, t3)

    # ---- final fine sweep of the outer angle on the best inner set ----
    if best_inner is not None:
        pts, _ = _all_points(best_inner)
        fine = np.linspace(0.0, math.pi / 3.0, 301)
        sF, cF, tF = best_side, best_center, best_angle
        for ang in fine:
            b = _bounding_side(pts, ang, np.array(best_center))
            if b < sF - 1e-12:
                sF, tF = b, ang
        _consider(best_inner, best_center, sF, tF)

    # ---- fallback ----
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, math.degrees(best_angle)
