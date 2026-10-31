"""
Hexagon packing: minimize outer hexagon side length for N=11 unit hexagons.

Strategy:
1. Enumerate all 11-cell polyhexes (cluster topologies on the hex lattice).
2. For each polyhex, compute the minimal enclosing regular hexagon (side length S).
3. Rank polyhexes by S; keep the top K.
4. For the top candidates, run continuous optimization (positions, angles, outer
   hexagon center/angle) to refine the packing.
5. Return the best valid result.
"""

import numpy as np
import time
import math
from itertools import product
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY = 8.0
EPS = 1e-9

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def hexagon_vertices(cx, cy, s, angle_deg):
    """6 vertices of a regular hexagon, side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def axial_to_cartesian(q, r):
    """Axial hex coordinates -> Cartesian (center of unit hexagon)."""
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return (x, y)


def cartesian_to_axial(x, y):
    """Cartesian -> nearest axial hex coords."""
    r = y / 1.5
    q = x / math.sqrt(3) - r / 2.0
    # Round to nearest integer using hex rounding
    rq = round(q)
    rr = round(r)
    rs = round(-q - r)
    # Fix rounding errors
    dq = abs(rq - q)
    dr = abs(rr - r)
    ds = abs(rs - (-q - r))
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return (rq, rr)


def hex_neighbors(q, r):
    """6 neighbors of (q,r) in axial coords."""
    return [(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)]


# ---------------------------------------------------------------------------
# Polyhex enumeration (BFS / backtracking)
# ---------------------------------------------------------------------------

def _normalize(cells):
    """Translate cells so the lexicographically smallest is at origin, then canonicalize."""
    cells = sorted(cells)
    min_q = min(c[0] for c in cells)
    min_r = min(c[1] for c in cells)
    # Actually we want a proper canonical form: rotate/reflection
    # For simplicity, just translate so min is at (0,0) and sort
    norm = tuple(sorted((q - min_q, r - min_r) for q, r in cells))
    return norm


def _canonical(cells):
    """Canonical form under 6 rotations + 6 reflections of the hex lattice."""
    cells = list(cells)
    best = None
    for rot in range(6):
        for reflect in [False, True]:
            transformed = []
            for q, r in cells:
                if reflect:
                    # Reflect: (q, r) -> (r, q) is one reflection
                    q, r = r, q
                # Rotate by 60 degrees: (q, r) -> (-r, q + r)
                for _ in range(rot):
                    q, r = -r, q + r
                transformed.append((q, r))
            norm = _normalize(transformed)
            if best is None or norm < best:
                best = norm
    return best


def enumerate_polyhexes(n_cells):
    """Enumerate all polyhexes of size n_cells. Returns list of canonical cell sets."""
    if n_cells == 1:
        return [((0, 0),)]
    
    seen = set()
    results = []
    
    # Start from all (n-1)-cell polyhexes
    prev = enumerate_polyhexes(n_cells - 1)
    
    for cells in prev:
        cells = list(cells)
        # Find all boundary edges (empty neighbors)
        cell_set = set(cells)
        candidates = set()
        for q, r in cells:
            for nq, nr in hex_neighbors(q, r):
                if (nq, nr) not in cell_set:
                    candidates.add((nq, nr))
        
        for cand in candidates:
            new_cells = tuple(sorted(cells + [cand]))
            canon = _canonical(new_cells)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
    
    return results


# ---------------------------------------------------------------------------
# Minimal enclosing hexagon for a set of points
# ---------------------------------------------------------------------------

def min_enclosing_hex_side(pts, beta):
    """
    Given points (N,2) and outer hexagon rotation beta (radians),
    find the minimum side length S of a regular hexagon with that rotation
    that contains all points. Returns (S, center_x, center_y) or (inf, None, None).
    """
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    
    # 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    normals = np.array(normals)  # (6, 2)
    
    # For a hexagon with center (cx, cy) and apothem D = S * sqrt(3)/2,
    # each point (px, py) must satisfy:
    #   dot((px-cx, py-cy), n_k) <= D  for all k
    #   dot((px-cx, py-cy), -n_k) <= D  for all k
    # i.e., |dot((px-cx, py-cy), n_k)| <= D
    #
    # We want to minimize D (hence S = 2D/sqrt(3)) over (cx, cy, D).
    # This is a linear program:
    #   minimize D
    #   subject to: n_k . (p_i - c) <= D  for all i, k
    #               -n_k . (p_i - c) <= D  for all i, k
    #
    # Rewrite: n_k . p_i - n_k . c <= D
    #          -n_k . p_i + n_k . c <= D
    
    n_pts = len(pts)
    # Variables: cx, cy, D
    # Objective: [0, 0, 1]
    # Constraints:
    #   For each point i and normal k:
    #     n_kx * px_i + n_ky * py_i - n_kx * cx - n_ky * cy - D <= 0
    #   For each point i and normal k:
    #    -n_kx * px_i - n_ky * py_i + n_kx * cx + n_ky * cy - D <= 0
    
    A_ub = []
    b_ub = []
    for px, py in pts:
        for nx, ny in normals:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj=np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.status == 0:
            D = res.x[2]
            S = 2.0 * D / math.sqrt(3.0)
            return S, res.x[0], res.x[1]
    except Exception:
        pass
    
    return float('inf'), None, None


def min_enclosing_hex_side_fast(pts):
    """
    Find minimal enclosing hexagon by trying a set of beta angles and
    refining the best one. Returns (S, beta, cx, cy).
    """
    best_S = float('inf')
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Try 12 evenly spaced angles in [0, pi/6)
    for i in range(12):
        beta = i * (math.pi / 6.0) / 12.0
        S, cx, cy = min_enclosing_hex_side(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_cx = cx if cx is not None else 0.0
            best_cy = cy if cy is not None else 0.0
    
    # Refine around the best beta
    if best_S < float('inf'):
        try:
            def obj(b):
                S, _, _ = min_enclosing_hex_side(pts, b)
                return S
            res = minimize_scalar(obj, bounds=(max(0, best_beta - 0.05), min(math.pi/6, best_beta + 0.05)),
                                  method='bounded', options={'maxiter': 50, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_cx, best_cy = min_enclosing_hex_side(pts, best_beta)
                if best_cx is None:
                    best_cx, best_cy = 0.0, 0.0
        except Exception:
            pass
    
    return best_S, best_beta, best_cx, best_cy


# ---------------------------------------------------------------------------
# Overlap check (SAT) and validity
# ---------------------------------------------------------------------------

def polygons_overlap(v1, v2, tol=1e-9):
    """SAT overlap check for two convex polygons."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check no overlaps and all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i + 1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            if polygons_overlap(v1, v2, tol=1e-7):
                return False
    
    # Check inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    n_outer = len(outer_verts)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(n_outer):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_outer]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i + 2) % n_outer] - v1)
                if d1 * d2 < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


# ---------------------------------------------------------------------------
# Continuous refinement
# ---------------------------------------------------------------------------

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length containing all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    normals = np.array([[np.cos(oa + np.radians(30 + 60 * k)), np.sin(oa + np.radians(30 + 60 * k))]
                        for k in range(6)])
    
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    all_verts_rel = all_verts - np.array([ox, oy])
    all_normals = np.vstack([normals, -normals])
    supports = np.max(all_verts_rel @ all_normals.T, axis=1)
    
    min_S = np.max(supports) / (np.sqrt(3.0) / 2.0)
    return max(min_S, 0.0)


def overlap_penalty(inner_hexagons, strength=500.0):
    """Smooth penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(inner_hexagons)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            min_d = 2.0 * np.sqrt(3.0) - 0.1
            if d < min_d:
                excess = min_d - d
                penalty += strength * excess ** 2
    return penalty


def objective(params, deadline):
    """Objective: outer side length + overlap penalty."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = np.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = np.degrees(params[3 * n + 2]) % 360
    
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    penalty = overlap_penalty(inner)
    
    return S + penalty


def refine_cluster(cells, beta_init, cx_init, cy_init, deadline):
    """
    Refine a polyhex cluster (list of axial coords) via continuous optimization.
    Returns best (inner, center, S, angle) found.
    """
    # Initial positions: axial -> cartesian, all angles 0
    pts = np.array([axial_to_cartesian(q, r) for q, r in cells])
    
    # Build initial params
    params = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(pts):
        params[3 * i] = px
        params[3 * i + 1] = py
        params[3 * i + 2] = 0.0
    params[3 * N] = cx_init
    params[3 * N + 1] = cy_init
    params[3 * N + 2] = beta_init
    
    best = None
    best_score = float('inf')
    
    def try_result(p):
        nonlocal best, best_score
        inner = []
        for i in range(N):
            cx = p[3 * i]
            cy = p[3 * i + 1]
            ang = np.degrees(p[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        ox = p[3 * N]
        oy = p[3 * N + 1]
        outer_angle = np.degrees(p[3 * N + 2]) % 360
        
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        
        if check_validity(inner, (ox, oy), S, outer_angle):
            if S < best_score:
                best_score = S
                best = (inner, (ox, oy), S, outer_angle)
    
    try_result(params)
    
    # L-BFGS-B refinement
    if time.time() < deadline:
        try:
            res = minimize(objective, params, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Nelder-Mead on outer params
    if best is not None and time.time() < deadline:
        try:
            inner_fixed = best[0]
            
            def outer_obj(p_outer):
                ox, oy, oa = p_outer
                return compute_outer_side_length(inner_fixed, (ox, oy), np.degrees(oa) % 360)
            
            outer_ic = np.array([best[1][0], best[1][1], np.radians(best[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    return best


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = START_TIME + TIME_LIMIT - SAFETY
    
    if time.time() > deadline:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_result = None
    best_score = float('inf')
    
    # --- Phase 1: Enumerate polyhexes and compute minimal enclosing hexagons ---
    if time.time() < deadline - 60:
        try:
            polyhexes = enumerate_polyhexes(N)
            # Sort by minimal enclosing hexagon side length
            scored = []
            for cells in polyhexes:
                if time.time() > deadline - 60:
                    break
                pts = np.array([axial_to_cartesian(q, r) for q, r in cells])
                S, beta, cx, cy = min_enclosing_hex_side_fast(pts)
                scored.append((S, beta, cx, cy, cells))
            
            scored.sort(key=lambda x: x[0])
            # Keep top K
            K = 8
            top_candidates = scored[:K]
        except Exception:
            top_candidates = []
    else:
        top_candidates = []
    
    # --- Phase 2: Refine top candidates ---
    for S_init, beta, cx, cy, cells in top_candidates:
        if time.time() > deadline - 20:
            break
        try:
            result = refine_cluster(list(cells), beta, cx, cy, deadline)
            if result is not None:
                inner, center, S, angle = result
                if S < best_score:
                    best_score = S
                    best_result = (inner, center, S, angle)
        except Exception:
            pass
    
    # --- Phase 3: Differential Evolution (if time remains) ---
    if time.time() < deadline - 30:
        try:
            bounds = []
            for i in range(N):
                bounds.append((-8, 8))
                bounds.append((-8, 8))
                bounds.append((0, 2 * np.pi))
            bounds.append((-5, 5))
            bounds.append((-5, 5))
            bounds.append((0, 2 * np.pi))
            
            # Use the best result as a starting point if available
            if best_result is not None:
                x0 = np.zeros(3 * N + 3)
                for i, (cx, cy, ang) in enumerate(best_result[0]):
                    x0[3 * i] = cx
                    x0[3 * i + 1] = cy
                    x0[3 * i + 2] = np.radians(ang)
                x0[3 * N] = best_result[1][0]
                x0[3 * N + 1] = best_result[1][1]
                x0[3 * N + 2] = np.radians(best_result[3])
            else:
                x0 = None
            
            de_result = differential_evolution(
                objective, bounds, args=(deadline,),
                maxiter=30, popsize=12, tol=1e-6,
                mutation=(0.3, 1.0), recombination=0.9,
                seed=42, polish=False,
                update_interval=5,
                callback=lambda x, conv: (time.time() > deadline)
            )
            
            if de_result.fun < float('inf'):
                inner = []
                for i in range(N):
                    cx = de_result.x[3 * i]
                    cy = de_result.x[3 * i + 1]
                    ang = np.degrees(de_result.x[3 * i + 2]) % 360
                    inner.append([cx, cy, ang])
                ox = de_result.x[3 * N]
                oy = de_result.x[3 * N + 1]
                outer_angle = np.degrees(de_result.x[3 * N + 2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # --- Phase 4: Final polish ---
    if best_result is not None and time.time() < deadline - 10:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = np.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < float('inf'):
                inner = []
                for i in range(N):
                    cx = res.x[3 * i]
                    cy = res.x[3 * i + 1]
                    ang = np.degrees(res.x[3 * i + 2]) % 360
                    inner.append([cx, cy, ang])
                ox = res.x[3 * N]
                oy = res.x[3 * N + 1]
                outer_angle = np.degrees(res.x[3 * N + 2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        hex_positions = [(0.0, 0.0)]
        for k in range(6):
            angle = np.radians(60 * k)
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            hex_positions.append((x, y))
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2 * math.sqrt(3) * math.cos(angle)
            y = 2 * math.sqrt(3) * math.sin(angle)
            hex_positions.append((x, y))
            if len(hex_positions) >= N:
                break
        hex_positions = hex_positions[:N]
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result


# EVOLVE-BLOCK-END
