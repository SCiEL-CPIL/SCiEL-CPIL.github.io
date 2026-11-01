"""Hexagonal packing: rigid composite with contact graph, then semi-rigid and full joint refinement."""
import math
import time
import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
MIN_DIST = SQRT3  # minimum center distance for unit hexagons

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def hex_vertices_local(side, angle_rad):
    """Vertices of a unit hexagon centered at origin with given rotation."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([side * np.cos(thetas), side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    # For a regular hexagon with side s and angle a, the support function in direction theta
    # is s * (cos(a - theta) + sqrt(3)*sin(a - theta)) for theta in [a-30, a+30] mod 60
    # Actually, let's use the vertex-based approach which is more general.
    # The outer hexagon has 6 vertices. For each inner hexagon vertex, we need it to be
    # inside the outer hexagon.
    # A point (px, py) is inside a regular hexagon centered at (ox, oy) with side s and angle a
    # iff for all 6 edge normals, the projection is within [-s*sqrt(3)/2, s*sqrt(3)/2] after
    # rotating by -a.
    # Actually, the inradius of a regular hexagon with side s is s*sqrt(3)/2.
    # The distance from center to each edge is s*sqrt(3)/2.
    # So a point is inside iff its distance to each edge (in the inward normal direction) <= s*sqrt(3)/2.
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # The 6 edge normals of a hexagon with angle 0 (flat top) point at angles 30, 90, 150, 210, 270, 330
    # Wait, let me reconsider. A regular hexagon with vertices at angles 0, 60, 120, ... has
    # edges between consecutive vertices. The outward normals to the edges point at angles 30, 90, 150, ...
    # The inradius is s*sqrt(3)/2.
    
    # For a hexagon rotated by angle a, the edge normals are at angles a + 30 + k*60 for k=0..5
    # A point p is inside iff for all k: (p - center) · n_k <= s * sqrt(3) / 2
    # where n_k is the unit outward normal to edge k.
    
    normals = np.array([[math.cos(math.radians(outer_angle_deg + 30 + k * 60)),
                         math.sin(math.radians(outer_angle_deg + 30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            d = v - np.array([ox, oy])
            projs = normals @ d
            m = projs.max()
            if m > max_proj:
                max_proj = m
    
    # max_proj is the maximum distance from center to any vertex in the direction of an edge normal
    # We need s * sqrt(3)/2 >= max_proj, so s >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth version using log-sum-exp for the max."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    
    normals = np.array([[math.cos(math.radians(outer_angle_deg + 30 + k * 60)),
                         math.sin(math.radians(outer_angle_deg + 30 + k * 60))] for k in range(6)])
    
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            d = v - np.array([ox, oy])
            projs = normals @ d
            all_projs.extend(projs.tolist())
    
    P = np.array(all_projs)
    if P.size == 0:
        return 0.0
    
    m = P.max()
    if m < -1e6:
        return 0.0
    
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """Separating axis test for two convex polygons."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Check that all inner hexagons are pairwise disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=MIN_DIST - 0.001):
    """Penalty for center distances below minimum."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_lattice_config():
    """Build a hexagonal lattice configuration for 11 hexagons.
    Returns relative positions (n, 2) and contact edges."""
    # Axial coordinates for a hexagonal arrangement of 11 cells:
    # Center (0,0), ring of 6, and 4 more in a compact shape
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    # Convert to Cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
    rel_pos = np.zeros((N, 2))
    for i, (q, r) in enumerate(axial):
        rel_pos[i, 0] = 1.5 * q
        rel_pos[i, 1] = SQRT3 * (r + q / 2.0)
    
    # Contact edges: pairs at distance MIN_DIST
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(rel_pos[i, 0] - rel_pos[j, 0], rel_pos[i, 1] - rel_pos[j, 1])
            if abs(d - MIN_DIST) < 0.01:
                edges.append((i, j))
    
    return rel_pos, edges, axial

def rigid_composite_objective(params, rel_pos, edges, beta=50.0):
    """
    Objective for rigid composite: 6 DOF (cx, cy, angle, ox, oy, oa)
    where (cx, cy, angle) define the rigid body transform and (ox, oy, oa) define the outer hexagon.
    """
    cx, cy, ca, ox, oy, oa = params[0], params[1], params[2], params[3], params[4], params[5]
    cos_ca = math.cos(ca)
    sin_ca = math.sin(ca)
    
    # Transform relative positions to world coordinates
    inner = []
    for i in range(N):
        rx, ry = rel_pos[i, 0], rel_pos[i, 1]
        wx = cx + rx * cos_ca - ry * sin_ca
        wy = cy + rx * sin_ca + ry * cos_ca
        wa = math.degrees(ca)  # All hexagons have the same rotation
        inner.append([wx, wy, wa])
    
    side = compute_required_side_smooth(inner, (ox, oy), oa, beta=beta)
    return side

def optimize_composite_rigid(rel_pos, edges, deadline):
    """
    Optimize the rigid composite: 6 DOF (cx, cy, angle, ox, oy, oa).
    """
    cx0 = float(np.mean(rel_pos[:, 0]))
    cy0 = float(np.mean(rel_pos[:, 1]))
    
    def obj(p, beta=50.0):
        return rigid_composite_objective(p, rel_pos, edges, beta=beta)
    
    def obj_exact(p):
        cx, cy, ca, ox, oy, oa = p
        cos_ca = math.cos(ca)
        sin_ca = math.sin(ca)
        inner = []
        for i in range(N):
            rx, ry = rel_pos[i, 0], rel_pos[i, 1]
            wx = cx + rx * cos_ca - ry * sin_ca
            wy = cy + rx * sin_ca + ry * cos_ca
            wa = math.degrees(ca)
            inner.append([wx, wy, wa])
        return compute_required_side_fast(inner, (ox, oy), oa)
    
    best_val = np.inf
    best_p = None
    
    # Multi-start over the rigid body angle
    for ca0 in np.arange(0, 2 * np.pi, np.pi / 6):
        if time.time() > deadline - 15:
            break
        p0 = np.array([cx0, cy0, ca0, cx0, cy0, 0.0])
        try:
            r1 = minimize(obj, p0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 800})
            r2 = minimize(obj_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 1000})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        # Fallback
        p0 = np.array([cx0, cy0, 0.0, cx0, cy0, 0.0])
        inner = [[rel_pos[i, 0], rel_pos[i, 1], 0.0] for i in range(N)]
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    cx, cy, ca, ox, oy, oa = best_p
    cos_ca = math.cos(ca)
    sin_ca = math.sin(ca)
    inner = []
    for i in range(N):
        rx, ry = rel_pos[i, 0], rel_pos[i, 1]
        wx = cx + rx * cos_ca - ry * sin_ca
        wy = cy + rx * sin_ca + ry * cos_ca
        wa = math.degrees(ca)
        inner.append([wx, wy, wa])
    
    return inner, (ox, oy), oa, best_val

def semi_rigid_optimize(inner, oc, oa, deadline):
    """
    Semi-rigid optimization: fix the contact graph structure but allow individual angles to vary.
    Optimize all positions and individual angles with a smooth objective.
    """
    def obj(p, beta=50.0):
        il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
        side = compute_required_side_smooth(il, (p[N*3], p[N*3+1]), p[N*3+2], beta=beta)
        pen = overlap_penalty(il)
        return side + 100.0 * pen
    
    def obj_exact(p):
        il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
        side = compute_required_side_fast(il, (p[N*3], p[N*3+1]), p[N*3+2])
        pen = overlap_penalty(il)
        return side + 500.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a
    p0[N*3] = oc[0]
    p0[N*3+1] = oc[1]
    p0[N*3+2] = oa
    
    # Try a few random perturbations of angles
    best_val = obj_exact(p0)
    best_p = p0.copy()
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        p_try = p0.copy()
        if attempt > 0:
            # Perturb angles
            for i in range(N):
                p_try[i*3+2] += np.random.uniform(-10, 10)
        try:
            r = minimize(obj, p_try, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            r2 = minimize(obj_exact, r.x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 2000})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    final_inner = [[float(best_p[i*3]), float(best_p[i*3+1]), float(best_p[i*3+2])] for i in range(N)]
    final_oc = (float(best_p[N*3]), float(best_p[N*3+1]))
    final_oa = float(best_p[N*3+2])
    final_side = compute_required_side_fast(final_inner, final_oc, final_oa)
    
    return final_inner, final_oc, final_oa, final_side

def full_joint_optimize(inner, oc, oa, deadline):
    """
    Full joint optimization with differential evolution for global search.
    """
    def obj_joint(p, beta=40.0):
        il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
        side = compute_required_side_smooth(il, (p[N*3], p[N*3+1]), p[N*3+2], beta=beta)
        pen = overlap_penalty(il)
        return side + 100.0 * pen
    
    def obj_joint_exact(p):
        il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
        side = compute_required_side_fast(il, (p[N*3], p[N*3+1]), p[N*3+2])
        pen = overlap_penalty(il)
        return side + 500.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a
    p0[N*3] = oc[0]
    p0[N*3+1] = oc[1]
    p0[N*3+2] = oa
    
    bounds = []
    for i in range(N):
        bounds.append((inner[i][0] - 3, inner[i][0] + 3))
        bounds.append((inner[i][1] - 3, inner[i][1] + 3))
        bounds.append((-30, 30))
    bounds.append((oc[0] - 2, oc[0] + 2))
    bounds.append((oc[1] - 2, oc[1] + 2))
    bounds.append((-30, 30))
    
    best_p = p0.copy()
    best_val = obj_joint_exact(p0)
    
    # Differential evolution
    try:
        de_result = differential_evolution(
            obj_joint, bounds,
            maxiter=30, popsize=12,
            tol=1e-5, seed=42,
            update_interval=1, polish=False
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_p = de_result.x
    except Exception:
        pass
    
    # Nelder-Mead polish
    try:
        nm_result = minimize(obj_joint_exact, best_p, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 3000})
        if nm_result.fun < best_val:
            best_val = nm_result.fun
            best_p = nm_result.x
    except Exception:
        pass
    
    final_inner = [[float(best_p[i*3]), float(best_p[i*3+1]), float(best_p[i*3+2])] for i in range(N)]
    final_oc = (float(best_p[N*3]), float(best_p[N*3+1]))
    final_oa = float(best_p[N*3+2])
    
    return final_inner, final_oc, final_oa

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    # Phase 1: Rigid composite optimization
    rel_pos, edges, axial = build_lattice_config()
    inner, oc, oa, side_val = optimize_composite_rigid(rel_pos, edges, deadline)
    
    best_inner = inner
    best_oc = oc
    best_oa = oa
    best_side = side_val
    
    # Phase 2: Semi-rigid angle refinement
    if time.time() < deadline - 40:
        semi_inner, semi_oc, semi_oa, semi_side = semi_rigid_optimize(
            best_inner, best_oc, best_oa, deadline)
        if check_disjoint(semi_inner, eps=1e-5):
            if semi_side < best_side:
                best_inner = semi_inner
                best_oc = semi_oc
                best_oa = semi_oa
                best_side = semi_side
    
    # Phase 3: Full joint optimization (final polish)
    if time.time() < deadline - 20:
        joint_inner, joint_oc, joint_oa = full_joint_optimize(
            best_inner, best_oc, best_oa, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_oc = joint_oc
                best_oa = joint_oa
                best_side = joint_side
    
    # Final verification and fallback
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        centers = [(1.5*s0*q, SQRT3*s0*(r+q/2.0)) for q, r in axial]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_oc = (cx0, cy0)
        best_oa = 0.0
        best_side = compute_required_side_fast(best_inner, best_oc, best_oa)
    
    return best_inner, best_oc, best_side, best_oa

# EVOLVE-BLOCK-END
