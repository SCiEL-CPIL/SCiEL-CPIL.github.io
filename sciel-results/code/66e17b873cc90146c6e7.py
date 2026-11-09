"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines SAT-based overlap detection + L-BFGS-B optimization (current) with
analytical minimal-enclosing-hexagon + honeycomb lattice initializations (inspiration).
"""
N = 12

import numpy as np
import time
import math
from scipy.optimize import minimize

SQRT3 = np.sqrt(3.0)

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_normals(angle_deg):
    """Unit outward face normals for a regular hexagon with given rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a + np.radians(60 * i + 30)),
         np.sin(a + np.radians(60 * i + 30))]
        for i in range(6)
    ])

def _hex_support(center, angle_deg, side, d):
    """Support function of a regular hexagon in direction d (unit vector)."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _sat_overlap_depth(verts_a, verts_b):
    """Minimum penetration depth between two convex polygons via SAT. 0 if no overlap."""
    min_depth = float('inf')
    all_normals = []
    for verts in [verts_a, verts_b]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                all_normals.append(n)
    for n in all_normals:
        proj_a = np.dot(verts_a, n)
        proj_b = np.dot(verts_b, n)
        min_a = np.min(proj_a); max_a = np.max(proj_a)
        min_b = np.min(proj_b); max_b = np.max(proj_b)
        if max_a <= min_b or max_b <= min_a:
            return 0.0
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
    return min_depth

def _containment_violation(centers, S, outer_angle_deg=0.0):
    """Total containment violation for all inner hexagons vs the outer hexagon."""
    outer_normals = _hex_normals(outer_angle_deg)
    outer_apothem = S * SQRT3 / 2.0
    violation = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        for n in outer_normals:
            h = _hex_support(np.array([cx, cy]), 0.0, 1.0, n)
            d = cx * n[0] + cy * n[1]
            needed = d + h
            if needed > outer_apothem:
                violation += (needed - outer_apothem) ** 2
    return violation

def _total_penalty(centers, S, outer_angle_deg=0.0):
    """Containment violation + overlap penalty between inner hexagons."""
    penalty = _containment_violation(centers, S, outer_angle_deg)
    overlap_penalty = 0.0
    for i in range(N):
        vi = _hex_verts(centers[i][0], centers[i][1], 1.0, 0.0)
        for j in range(i + 1, N):
            vj = _hex_verts(centers[j][0], centers[j][1], 1.0, 0.0)
            depth = _sat_overlap_depth(vi, vj)
            if depth > 0:
                overlap_penalty += depth ** 2
    return penalty

def _check_validity(centers, S, outer_angle_deg=0.0, tol=1e-7):
    """Check containment + mutual disjointness."""
    outer_normals = _hex_normals(outer_angle_deg)
    outer_apothem = S * SQRT3 / 2.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        for n in outer_normals:
            h = _hex_support(np.array([cx, cy]), 0.0, 1.0, n)
            d = cx * n[0] + cy * n[1]
            if d + h > outer_apothem + tol:
                return False
    for i in range(N):
        vi = _hex_verts(centers[i][0], centers[i][1], 1.0, 0.0)
        for j in range(i + 1, N):
            vj = _hex_verts(centers[j][0], centers[j][1], 1.0, 0.0)
            depth = _sat_overlap_depth(vi, vj)
            if depth > tol:
                return False
    return True

def _min_enclosing_hexagon(inner, inner_angles, inner_sides):
    """
    Find the minimal regular hexagon (center, side, angle) containing all inner
    hexagons. inner: list of (x, y, angle_deg). Returns (cx, cy, side, angle_deg).
    """
    all_verts = []
    for (cx, cy, ang), s in zip(inner, inner_sides):
        all_verts.append(_hex_verts(cx, cy, s, ang))
    all_verts = np.vstack(all_verts)

    best_side = float('inf')
    best_center = None
    best_angle = 0.0

    for phi_deg in np.linspace(0, 60, 1200):
        phi = np.radians(phi_deg)
        normals = np.array([
            [np.cos(phi + np.radians(60 * i + 30)), np.sin(phi + np.radians(60 * i + 30))]
            for i in range(6)
        ])
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / SQRT3

        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        if np.any(lo > hi + 1e-12):
            continue
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg

    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    return best_center[0], best_center[1], best_side, best_angle

def _honeycomb_positions(pattern, scale=1.0):
    """Generate honeycomb-lattice positions from a row-length pattern (sums to N)."""
    positions = []
    for j, row_len in enumerate(pattern):
        for i in range(row_len):
            x = (i + 0.5 * (j % 2)) * SQRT3 * scale
            y = j * 1.5 * scale
            positions.append(np.array([x, y]))
    # Center the configuration
    if positions:
        mean = np.mean(positions, axis=0)
        positions = [p - mean for p in positions]
    return positions[:N]

def _make_grid_init(scale=1.02):
    """Hexagonal lattice grid initialization."""
    u = np.array([1.5, SQRT3 / 2.0])
    v = np.array([-1.5, SQRT3 / 2.0])
    centers = []
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            centers.append(np.array([pos[0], pos[1]]))
    mean = np.mean(centers, axis=0)
    centers = [c - mean for c in centers]
    return centers

def _ring_init(scale=1.0):
    """Ring + center initialization."""
    centers = [np.array([0.0, 0.0])]
    for i in range(6):
        angle = np.radians(60 * i)
        r = scale * 2.0
        centers.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    for i in range(5):
        angle = np.radians(60 * i + 30)
        r = scale * 3.4
        centers.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    return centers

def _refine_local(centers, S, max_iter=2000, tol=1e-8):
    """Refine a configuration via L-BFGS-B to minimize total penalty."""
    x0 = np.array([c[0] for c in centers] + [c[1] for c in centers])

    def objective(params):
        c = np.array([[params[2 * i], params[2 * i + 1]] for i in range(N)])
        return _total_penalty(c, S)

    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        refined = np.array([[res.x[2 * i], res.x[2 * i + 1]] for i in range(N)])
        return refined, res.fun
    except Exception:
        return centers, _total_penalty(centers, S)

def _try_pack(centers_init, S, max_iter=1500):
    """Try to pack using given initial centers and outer side length S."""
    refined, penalty = _refine_local(centers_init, S, max_iter=max_iter)
    if _check_validity(refined, S):
        return True, refined
    return False, refined

def _best_enclosing_side(centers):
    """Compute the minimal enclosing hexagon side for a configuration."""
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    inner_angles = [0.0] * N
    inner_sides = [1.0] * N
    cx, cy, side, angle = _min_enclosing_hexagon(inner, inner_angles, inner_sides)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate diverse initializations: honeycomb patterns + grid + ring
    patterns = [
        [3, 4, 3, 2], [4, 3, 3, 2], [2, 3, 4, 3], [3, 3, 4, 2],
        [4, 4, 2, 2], [2, 2, 4, 4], [5, 3, 2, 2], [2, 2, 3, 5],
        [6, 3, 3], [3, 3, 6], [4, 4, 4], [2, 4, 4, 2],
        [4, 2, 4, 2], [2, 4, 2, 4], [3, 5, 4], [4, 5, 3],
        [5, 4, 3], [3, 4, 5],
    ]
    inits = []
    for pattern in patterns:
        if sum(pattern) != 12 or len(pattern) > 6:
            continue
        inits.append(_honeycomb_positions(pattern, 1.0))
    for scale in [1.0, 1.02, 1.05]:
        inits.append(_make_grid_init(scale))
    for scale in [1.0, 0.98, 1.02]:
        inits.append(_ring_init(scale))

    best_S = float('inf')
    best_centers = None

    # First pass: find a rough upper bound
    for init in inits:
        if time.time() > deadline - 70.0:
            break
        for S_try in [4.0, 4.5, 5.0, 5.5, 6.0]:
            if time.time() > deadline - 65.0:
                break
            success, refined = _try_pack(init, S_try, max_iter=800)
            if success and S_try < best_S:
                best_S = S_try
                best_centers = refined

    if best_centers is None:
        init = _honeycomb_positions([3, 4, 3, 2], 1.05)
        S_fallback = 6.0
        success, refined = _try_pack(init, S_fallback, max_iter=1500)
        if success:
            best_S = S_fallback
            best_centers = refined
        else:
            best_S = 6.0
            best_centers = init
            best_S = _best_enclosing_side(best_centers)

    # Binary search for minimum valid S (more iterations, tighter)
    S_lo = 3.5
    S_hi = best_S

    if time.time() < deadline - 35.0:
        for iteration in range(14):
            if time.time() > deadline - 30.0:
                break
            S_mid = (S_lo + S_hi) / 2.0
            success = False
            best_refined = None
            for init in inits[:6]:
                if time.time() > deadline - 25.0:
                    break
                s, r = _try_pack(init, S_mid, max_iter=1000)
                if s:
                    success = True
                    best_refined = r
                    break
            if success:
                S_hi = S_mid
                best_S = S_mid
                best_centers = best_refined
            else:
                S_lo = S_mid

    # Final high-precision polish
    if time.time() < deadline - 12.0:
        refined, penalty = _refine_local(best_centers, best_S, max_iter=4000, tol=1e-12)
        if _check_validity(refined, best_S, tol=1e-9):
            best_centers = refined

    # Compute the true minimal enclosing hexagon for the final configuration
    inner = [[float(c[0]), float(c[1]), 0.0] for c in best_centers]
    inner_angles = [0.0] * N
    inner_sides = [1.0] * N
    cx, cy, final_S, angle = _min_enclosing_hexagon(inner, inner_angles, inner_sides)

    # Safety margin for numerical precision
    final_S = final_S * 1.0001 + 1e-6

    return inner, (float(cx), float(cy)), float(final_S), float(angle)
