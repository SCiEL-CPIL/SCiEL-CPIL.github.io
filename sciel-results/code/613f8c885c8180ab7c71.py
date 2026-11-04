"""Structural rewrite: Nelder-Mead local search with SAT-based collision penalty.

Strategy:
- Use a smooth SAT-based overlap penalty (sum of squared penetration depths)
  which gives a much better gradient signal than hard validity checks.
- Nelder-Mead (scipy.optimize.minimize) for local refinement.
- Multiple random restarts with diverse seeds.
- Fast bounding hexagon computation.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta, center=None):
    """Regular hexagon, side 1, circumradius 1, rotated by theta, translated by center."""
    if center is None:
        center = np.zeros(2)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)]) + center

def _hex_edge_normals(theta):
    """6 outward edge normals of a regular hexagon oriented by theta."""
    return np.column_stack([
        np.cos(theta + np.pi / 6.0 + np.arange(6) * np.pi / 3.0),
        np.sin(theta + np.pi / 6.0 + np.arange(6) * np.pi / 3.0)
    ])

def _project_polygon(pts, axis):
    """Project polygon vertices onto axis; return (min, max)."""
    proj = pts @ axis
    return proj.min(), proj.max()

def _separating_axis_check(A, B):
    """SAT check. Returns (overlap, min_depth, best_axis).
    overlap=True means they intersect.
    min_depth < 0 means penetration depth.
    """
    min_depth = float('inf')
    best_axis = None
    for poly in (A, B):
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            nl = np.linalg.norm(axis)
            if nl < 1e-12:
                continue
            axis /= nl
            amin, amax = _project_polygon(A, axis)
            bmin, bmax = _project_polygon(B, axis)
            sep = min(amax - bmin, bmax - amin)
            if sep < 0:
                return True, sep, axis  # definitely overlapping
            if sep < min_depth:
                min_depth = sep
                best_axis = axis
    if min_depth < 0:
        return True, min_depth, best_axis
    return False, min_depth, best_axis

def _outer_hex_side(pts, theta):
    """Compute side length of smallest regular hexagon (orientation theta) containing pts."""
    normals = _hex_edge_normals(theta)
    P = pts @ normals.T  # shape (n, 6)
    # For each face, the extent is max(P[:,f]) - min(P[:,f])
    # The side length = max over faces of extent / sqrt(3)
    extents = P.max(axis=0) - P.min(axis=0)
    return float(np.max(extents) / math.sqrt(3.0))

def _is_valid(centers, angles, outer_side, outer_theta, outer_center):
    """Check if all hexagons are non-overlapping and contained in outer hexagon."""
    EPS = 1e-7
    verts = [_hex_vertices(angles[k], centers[k]) for k in range(N)]
    
    # Pairwise SAT
    for i in range(N):
        for j in range(i + 1, N):
            overlap, _, _ = _separating_axis_check(verts[i], verts[j])
            if overlap:
                return False
    
    # Containment
    oc = np.array(outer_center)
    outer_verts = _hex_vertices(outer_theta, oc) * outer_side
    # Check each inner hex vertex is inside outer hexagon
    outer_normals = _hex_edge_normals(outer_theta)
    for i in range(N):
        for v in verts[i]:
            rel = v - oc
            proj = outer_normals @ rel
            if np.any(proj > outer_side * math.sqrt(3.0) / 2.0 + EPS):
                return False
    return True

# ---------------------------------------------------------------------------
# Smooth objective with SAT-based overlap penalty
# ---------------------------------------------------------------------------
def _objective(params, outer_theta):
    """Objective: bounding hex side + penalty for overlaps.
    params: array of shape (3*N,) = [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = params[0::3].copy()
    angles = params[1::3].copy()
    
    verts = [_hex_vertices(angles[k], centers[k]) for k in range(N)]
    
    # Bounding hex side
    all_pts = np.vstack(verts)
    side = _outer_hex_side(all_pts, outer_theta)
    
    # Overlap penalty: sum of squared penetration depths
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            overlap, depth, _ = _separating_axis_check(verts[i], verts[j])
            if overlap:
                penalty += depth * depth
    
    # Containment penalty
    oc = np.zeros(2)  # We optimize outer center separately
    outer_normals = _hex_edge_normals(outer_theta)
    contain_penalty = 0.0
    for i in range(N):
        for v in verts[i]:
            rel = v - oc
            proj = outer_normals @ rel
            exceed = np.maximum(proj - outer_side * math.sqrt(3.0) / 2.0, 0.0)
            contain_penalty += np.sum(exceed * exceed)
    
    return side + 50.0 * penalty + 10.0 * contain_penalty

def _objective_with_outer(params, outer_theta, outer_center):
    """Objective for fixed outer center."""
    centers = params[0::3].copy()
    angles = params[1::3].copy()
    
    verts = [_hex_vertices(angles[k], centers[k]) for k in range(N)]
    
    all_pts = np.vstack(verts)
    # Shift to outer center frame
    oc = np.array(outer_center)
    all_pts_shifted = all_pts - oc
    
    side = _outer_hex_side(all_pts_shifted, outer_theta)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            overlap, depth, _ = _separating_axis_check(verts[i], verts[j])
            if overlap:
                penalty += depth * depth
    
    outer_normals = _hex_edge_normals(outer_theta)
    contain_penalty = 0.0
    for i in range(N):
        for v in verts[i]:
            rel = v - oc
            proj = outer_normals @ rel
            exceed = np.maximum(proj - side * math.sqrt(3.0) / 2.0, 0.0)
            contain_penalty += np.sum(exceed * exceed)
    
    return side + 50.0 * penalty + 10.0 * contain_penalty

# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _seed_hex_ring():
    """1 hex in center, 6 in ring, 5 more in second ring."""
    d = 2.0 * math.sqrt(3.0) + 0.01
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
              (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
    pts = np.array([i * a1 + j * a2 for (i, j) in coords])
    return pts - pts.mean(axis=0)

def _seed_trilattice():
    """Triangular lattice arrangement."""
    d = 2.0 * math.sqrt(3.0) + 0.01
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    # Pick 12 closest to center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    pts = pts[idx]
    return pts - pts.mean(axis=0)

def _seed_random():
    pts = np.array([[random.gauss(0, 3.0), random.gauss(0, 3.0)] for _ in range(N)])
    return pts - pts.mean(axis=0)

def _seed_spiral():
    """Spiral arrangement."""
    pts = []
    for k in range(N):
        r = 0.5 + k * 0.6
        theta = k * 0.8
        pts.append([r * math.cos(theta), r * math.sin(theta)])
    pts = np.array(pts)
    return pts - pts.mean(axis=0)

# ---------------------------------------------------------------------------
# Nelder-Mead local optimization
# ---------------------------------------------------------------------------
def _nelder_mead_optimize(centers, angles, outer_theta, outer_center, max_iters=500, time_limit=10.0):
    """Run Nelder-Mead to minimize the objective."""
    x0 = np.zeros(3 * N)
    for k in range(N):
        x0[3 * k] = centers[k][0]
        x0[3 * k + 1] = centers[k][1]
        x0[3 * k + 2] = angles[k]
    
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective_with_outer(x, outer_theta, outer_center)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-8})
    except Exception:
        return centers.copy(), angles.copy(), None
    
    best_centers = np.zeros((N, 2))
    best_angles = np.zeros(N)
    for k in range(N):
        best_centers[k][0] = res.x[3 * k]
        best_centers[k][1] = res.x[3 * k + 1]
        best_angles[k] = res.x[3 * k + 2]
    
    return best_centers, best_angles, res.fun

# ---------------------------------------------------------------------------
# Refine outer hexagon parameters
# ---------------------------------------------------------------------------
def _refine_outer(centers, angles):
    """Find best outer hexagon (center, angle) for given inner config."""
    all_pts = np.vstack([_hex_vertices(angles[k], centers[k]) for k in range(N)])
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for ang in np.linspace(0.0, math.pi / 3.0, 181):
        normals = _hex_edge_normals(ang)
        P = all_pts @ normals.T
        c_normal = (P.max(axis=0) + P.min(axis=0)) / 2.0
        c_world = normals.T @ c_normal
        extents = P.max(axis=0) - c_normal
        side = 2.0 * float(np.max(extents)) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = (float(c_world[0]), float(c_world[1]))
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    random.seed(123)
    np.random.seed(123)
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle
    
    seed_funcs = [_seed_hex_ring, _seed_trilattice, _seed_random, _seed_spiral]
    
    # First pass: quick Nelder-Mead on each seed
    for si, seed_func in enumerate(seed_funcs):
        if time.time() > DEADLINE - 20:
            break
        
        centers = seed_func()
        angles = np.zeros(N)
        
        # Get initial outer estimate
        side0, center0, angle0 = _refine_outer(centers, angles)
        
        # Nelder-Mead optimization
        bc, ba, _ = _nelder_mead_optimize(centers, angles, angle0, center0,
                                           max_iters=300, time_limit=8.0)
        
        # Refine outer
        side, center, angle = _refine_outer(bc, ba)
        
        _consider(bc, ba, side, center, angle)
    
    # Second pass: more thorough optimization on best configurations
    if best_centers is not None and time.time() < DEADLINE - 30:
        # Re-optimize from best
        side, center, angle = _refine_outer(best_centers, best_angles)
        bc, ba, _ = _nelder_mead_optimize(best_centers, best_angles, angle, center,
                                           max_iters=500, time_limit=15.0)
        side2, center2, angle2 = _refine_outer(bc, ba)
        _consider(bc, ba, side2, center2, angle2)
        
        # Try perturbing the best and re-optimizing
        for restart in range(3):
            if time.time() > DEADLINE - 15:
                break
            pert_centers = best_centers + np.random.normal(0, 0.3, (N, 2))
            pert_angles = best_angles + np.random.normal(0, 0.1, N)
            side_p, center_p, angle_p = _refine_outer(pert_centers, pert_angles)
            bc2, ba2, _ = _nelder_mead_optimize(pert_centers, pert_angles, angle_p, center_p,
                                                  max_iters=300, time_limit=8.0)
            side3, center3, angle3 = _refine_outer(bc2, ba2)
            _consider(bc2, ba2, side3, center3, angle3)
    
    # Final fine outer angle sweep
    if best_centers is not None:
        side, center, angle = _refine_outer(best_centers, best_angles)
        _consider(best_centers, best_angles, side, center, angle)
    
    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]),
                      float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
