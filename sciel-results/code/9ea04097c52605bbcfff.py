"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.

Structural approach: the best packings are rigid clusters of unit hexagons sitting
on a lattice. We enumerate many discrete lattice arrangements (hex + triangular
lattice, full 11-blob and 11-subsets of the 19-blob), and for each we solve the
rigid-body placement (global translation + rotation of the cluster relative to the
container) in a tiny 3D space with Nelder-Mead. This avoids the 36D continuous
local-minima trap of pure L-BFGS while still refining each candidate to high
precision. A final L-BFGS polish lets individual hexagons break free.
"""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _normals_from_verts(v):
    n = len(v)
    out = []
    for i in range(n):
        e = v[(i+1) % n] - v[i]
        nn = np.array([-e[1], e[0]])
        ln = np.linalg.norm(nn)
        if ln > 1e-12:
            out.append(nn / ln)
    return out

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = _normals_from_verts(v1) + _normals_from_verts(v2)
    for axis in axes:
        p1 = np.dot(v1, axis); p2 = np.dot(v2, axis)
        if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
            return False, axis
    return True, None

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    c = np.array([ox, oy])
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        rel = verts - c
        for n in normals:
            d = np.dot(rel, n).max()
            if d > 0:
                S_needed = 2.0 * d / np.sqrt(3.0)
                if S_needed > min_S:
                    min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle, tol=1e-7):
    """No overlaps between inner hexagons and all inside the outer hexagon."""
    ox, oy = outer_center
    for i in range(len(inner_hexagons)):
        cx1, cy1, a1 = inner_hexagons[i]
        v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
        for j in range(i+1, len(inner_hexagons)):
            cx2, cy2, a2 = inner_hexagons[j]
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=tol)
            if overlap:
                return False
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    on = _normals_from_verts(outer_verts)
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        rel = verts - np.array([ox, oy])
        for n in on:
            if np.dot(rel, n).max() > outer_side * np.sqrt(3.0)/2.0 + tol:
                return False
    return True

# ---------------------------------------------------------------------------
# Rigid-body placement of a fixed cluster of unit hexagons
# ---------------------------------------------------------------------------
def _rigid_obj(p, base, base_verts, base_norms, cache):
    t = p[0]
    theta = p[1]
    oa = p[2]
    c = cache.get((t, theta, oa))
    if c is None:
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        centers = base @ R.T + np.array([t, 0.0])
        verts = base_verts @ R.T + np.array([t, 0.0])
        S = 0.0
        for n in base_norms:
            d = np.dot(verts, n).max() - np.dot(centers, n).max()
            if d > 0:
                Sv = 2.0 * d / np.sqrt(3.0)
                if Sv > S:
                    S = Sv
        oth = oa + np.radians(30)
        on = np.array([[np.cos(oth + np.radians(60*k)) for k in range(6)],
                       [np.sin(oth + np.radians(60*k)) for k in range(6)]])
        rel = verts - on
        for k in range(6):
            d = rel[:, k].max()
            if d > 0:
                Sv = 2.0 * d / np.sqrt(3.0)
                if Sv > S:
                    S = Sv
        c = S
        if len(cache) > 200000:
            cache.clear()
        cache[(t, theta, oa)] = c
    return c

def place_cluster(base, deadline):
    """Find best rigid placement (translation + rotation) of a cluster of unit hexagons."""
    base = np.asarray(base, dtype=float)
    # center cluster at origin
    base = base - base.mean(axis=0)
    base_verts = np.vstack([hexagon_vertices(cx, cy, 1.0, 0.0) for cx, cy in base])
    base_norms = _normals_from_verts(hexagon_vertices(0, 0, 1.0, 0.0))
    cache = {}
    obj = lambda p: _rigid_obj(p, base, base_verts, base_norms, cache)

    best = None
    bestS = float('inf')
    inits = [(0.0, 0.0, 0.0)]
    for a in (0.0, 30.0):
        for b in (0.0, 15.0, 45.0):
            inits.append((a, b, 0.0))
    for ic in inits:
        if time.time() > deadline - 2.0:
            break
        try:
            r = minimize(obj, np.array(ic), method='Nelder-Mead',
                         options={'maxiter': 60, 'xatol': 1e-8, 'fatol': 1e-10})
            S = r.fun
            if S < bestS:
                bestS = S
                best = (float(r.x[0]), float(r.x[1]), float(r.x[2]))
        except Exception:
            pass
    if best is None:
        return None
    t, theta, oa = best
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    centers = base @ R.T + np.array([t, 0.0])
    inner = [[float(cx), float(cy), float(np.degrees(theta) % 360.0)] for cx, cy in centers]
    S = compute_outer_side_length(inner, (0.0, 0.0), float(np.degrees(oa) % 360.0))
    return inner, (0.0, 0.0), S, float(np.degrees(oa) % 360.0)

# ---------------------------------------------------------------------------
# Discrete lattice arrangements
# ---------------------------------------------------------------------------
def hex_cluster_positions():
    """11 unit hexagons in a hexagonal lattice: center + ring1(6) + 4 of ring2."""
    pos = [(0.0, 0.0)]
    for k in range(6):
        a = np.radians(60*k)
        pos.append((np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)))
    for k in range(6):
        a = np.radians(60*k)
        pos.append((2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a)))
        if len(pos) >= N:
            break
    return pos[:N]

def triangular_blob_positions():
    """19 unit hexagons packed on a triangular lattice (3-ring blob)."""
    a = np.sqrt(3.0)
    pos = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i*a + (j % 2) * (a/2.0)
            y = j * 1.5
            if i*i + j*j <= 9:
                pos.append((x, y))
    return pos

def _is_valid_cluster(pos, tol=1e-6):
    for i in range(len(pos)):
        v1 = hexagon_vertices(pos[i][0], pos[i][1], 1.0, 0.0)
        for j in range(i+1, len(pos)):
            v2 = hexagon_vertices(pos[j][0], pos[j][1], 1.0, 0.0)
            ov, _ = polygons_overlap(v1, v2, tol=tol)
            if ov:
                return False
    return True

def _is_inside_cluster(pos, container, tol=1e-6):
    on = _normals_from_verts(hexagon_vertices(0, 0, container, 0.0))
    lim = container * np.sqrt(3.0)/2.0
    for cx, cy in pos:
        v = hexagon_vertices(cx, cy, 1.0, 0.0)
        for n in on:
            if np.dot(v, n).max() > lim + tol:
                return False
    return True

def build_candidates(deadline):
    """Enumerate discrete lattice arrangements (hex + triangular full blob + subsets)."""
    cands = []
    seen = set()

    def add(pos):
        key = tuple(round(x, 4) for x in np.array(pos).ravel())
        if key in seen:
            return
        if not _is_valid_cluster(pos, tol=1e-6):
            return
        seen.add(key)
        cands.append(pos)

    add(hex_cluster_positions())

    blob = triangular_blob_positions()
    add(blob[:N])  # first 11 of the 19-blob
    # symmetric 11-subsets of the 19-blob
    for comb in itertools.combinations(range(len(blob)), N):
        if time.time() > deadline - 30:
            break
        sub = [blob[i] for i in comb]
        add(sub)
    return cands

# ---------------------------------------------------------------------------
# L-BFGS polish (lets individual hexagons break free of the lattice)
# ---------------------------------------------------------------------------
def _lbfgs_objective(params, deadline):
    if time.time() > deadline:
        return float('inf')
    n = N
    inner = []
    for i in range(n):
        inner.append((params[3*i], params[3*i+1], np.degrees(params[3*i+2]) % 360))
    ox = params[3*n]; oy = params[3*n+1]
    oa = np.degrees(params[3*n+2]) % 360
    S = compute_outer_side_length(inner, (ox, oy), oa)
    penalty = 0.0
    for i in range(n):
        cx1, cy1, a1 = inner[i]
        for j in range(i+1, n):
            cx2, cy2, a2 = inner[j]
            d = np.hypot(cx1-cx2, cy1-cy2)
            if d < 2.5:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                ov, _ = polygons_overlap(v1, v2, tol=1e-8)
                if ov:
                    penalty += (2.0 - d)**2 * 100.0
    return S + penalty

def _params_from(inner, center, oa):
    p = np.zeros(3*N + 3)
    for i, (cx, cy, ang) in enumerate(inner):
        p[3*i] = cx; p[3*i+1] = cy; p[3*i+2] = np.radians(ang)
    p[3*N] = center[0]; p[3*N+1] = center[1]; p[3*N+2] = np.radians(oa)
    return p

def _result_from(params):
    inner = [[params[3*i], params[3*i+1], float(np.degrees(params[3*i+2]) % 360.0)]
             for i in range(N)]
    ox = params[3*N]; oy = params[3*N+1]
    oa = float(np.degrees(params[3*N+2]) % 360.0)
    S = compute_outer_side_length(inner, (ox, oy), oa)
    return inner, (ox, oy), S, oa

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    best_result = None
    best_score = float('inf')

    def consider(res):
        nonlocal best_result, best_score
        if res is None:
            return
        inner, center, S, oa = res
        if not check_validity(inner, center, S, oa):
            return
        if S < best_score:
            best_score = S
            best_result = (inner, center, S, oa)

    # 1) Discrete lattice enumeration + rigid placement (the structural core)
    cands = build_candidates(deadline)
    for pos in cands:
        if time.time() > deadline - 25:
            break
        consider(place_cluster(pos, deadline))

    # 2) L-BFGS polish on the best discrete arrangement (allow free movement)
    if best_result is not None and time.time() < deadline - 20:
        try:
            p0 = _params_from(best_result[0], best_result[1], best_result[3])
            r = minimize(_lbfgs_objective, p0, args=(deadline,), method='L-BFGS-B',
                         options={'maxiter': 250, 'ftol': 1e-13, 'gtol': 1e-11})
            consider(_result_from(r.x))
        except Exception:
            pass

    # 3) Rigid placement of the polished inner set (re-optimizes container)
    if best_result is not None and time.time() < deadline - 10:
        try:
            inner = best_result[0]
            base = np.array([[x, y] for x, y, _ in inner])
            consider(place_cluster(base, deadline))
        except Exception:
            pass

    # 4) Final L-BFGS polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            p0 = _params_from(best_result[0], best_result[1], best_result[3])
            r = minimize(_lbfgs_objective, p0, args=(deadline,), method='L-BFGS-B',
                         options={'maxiter': 120, 'ftol': 1e-15, 'gtol': 1e-13})
            consider(_result_from(r.x))
        except Exception:
            pass

    # Fallback (always a valid hex lattice configuration)
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_cluster_positions()]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)

    return best_result

# EVOLVE-BLOCK-END
