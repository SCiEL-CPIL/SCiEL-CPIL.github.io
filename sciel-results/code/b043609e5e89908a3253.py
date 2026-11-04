import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    """
    n = len(inner_params)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = SQRT3 / 2.0
    
    max_proj = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        projs = hex_v @ normals.T
        max_projs = np.max(projs, axis=0)
        required_sides = max_projs / apothem_factor
        max_proj = max(max_proj, np.max(required_sides))
    
    return max_proj

def _fast_overlap_penalty(inner_params, penalty_weight=100.0, threshold=0.02):
    """
    Fast overlap penalty using support functions for hexagons.
    Two convex polygons don't overlap iff there exists a separating axis.
    For regular hexagons, we can check 12 potential separating axes.
    """
    n = len(inner_params)
    penalty = 0.0
    
    # Precompute all vertices
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
    
    for i in range(n):
        for j in range(i+1, n):
            dx = inner_params[j, 0] - inner_params[i, 0]
            dy = inner_params[j, 1] - inner_params[i, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 16.0:
                continue
            
            sep = _poly_separation(verts[i], verts[j])
            if sep < threshold:
                penalty += penalty_weight * (threshold - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """Objective function: outer side length + overlap penalty."""
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _fast_overlap_penalty(inner_params)
    return side + overlap_pen

def _objective_smooth(params, outer_angle_deg, n, beta=30.0):
    """Smooth objective using soft-max for side computation."""
    inner_params = params.reshape(n, 3)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = SQRT3 / 2.0
    
    all_projs = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        projs = hex_v @ normals.T
        max_projs = np.max(projs, axis=0)
        required_sides = max_projs / apothem_factor
        all_projs.append(required_sides)
    
    all_required = np.array(all_projs)
    all_flat = all_required.flatten()
    m = np.max(all_flat)
    soft_max = m + np.log(np.sum(np.exp(all_flat - m))) / beta
    
    overlap_pen = _fast_overlap_penalty(inner_params)
    return soft_max + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """
    Return a good initial solution using a compact hexagonal arrangement.
    11 hexagons: center + 6 in first ring + 4 in second ring (partial).
    Use tight packing with minimal gaps.
    """
    centers = []
    # Center hexagon
    centers.append((0.0, 0.0))
    
    # First ring: 6 hexagons at distance sqrt(3) from center
    # For unit hexagons, adjacent centers are at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60 + 30)
        x = SQRT3 * np.cos(angle)
        y = SQRT3 * np.sin(angle)
        centers.append((x, y))
    
    # Second ring: 4 hexagons filling gaps
    # Place them at 30, 90, 150, 210 degrees at distance 3
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute tight bounding hexagon
    all_verts = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find minimum enclosing regular hexagon - try both orientations
    best_side = float('inf')
    best_angle = 30.0
    for outer_angle in [0, 30]:
        normals = np.column_stack([
            np.cos(np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle)),
            np.sin(np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle))
        ])
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        side = np.max(max_projs) / (SQRT3 / 2.0)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    side = best_side + 1e-10
    return inner, (0.0, 0.0), side, best_angle

def _local_refine(inner_params, outer_angle_deg, n, time_limit=20.0):
    """Use Nelder-Mead for local refinement."""
    start_time = time.time()
    init_params = inner_params.flatten()
    
    # Smooth objective first
    try:
        res_smooth = minimize(_objective_smooth, init_params, args=(outer_angle_deg, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res_smooth.fun < _objective(init_params, outer_angle_deg, n):
            init_params = res_smooth.x
    except Exception:
        pass
    
    # Exact objective for final polish
    if time.time() - start_time < time_limit:
        try:
            res = minimize(_objective, init_params, args=(outer_angle_deg, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
            
            if time.time() - start_time < time_limit:
                candidate = res.x.reshape(n, 3)
                if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                    return candidate
        except Exception:
            pass
    
    return inner_params

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    
    # Try multiple outer angles
    best_inner = None
    best_side = float('inf')
    best_outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    best_outer_angle = seed_angle
    
    # Multi-start optimization for each outer angle
    outer_angles_to_try = [0.0, 30.0]
    
    for outer_angle in outer_angles_to_try:
        remaining = DEADLINE - safety_margin - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Create multiple starting points
        starts = [best_inner.copy()]
        rng = np.random.RandomState(42)
        for _ in range(8):
            pert = rng.randn(n, 3) * 0.5
            pert[:, 2] *= 0.3  # smaller angle perturbations
            start = best_inner + pert
            start[:, 2] = np.mod(start[:, 2], 360)
            starts.append(start)
        
        for start_idx, start_params in enumerate(starts):
            remaining = DEADLINE - safety_margin - (time.time() - start_time)
            if remaining < 5:
                break
            
            init_params = start_params.flatten()
            try:
                # Smooth objective for faster convergence
                res = minimize(_objective_smooth, init_params, args=(outer_angle, n), 
                              method='Nelder-Mead', 
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
                
                if res.fun < best_side:
                    candidate = res.x.reshape(n, 3)
                    candidate_side = _compute_side(candidate, outer_angle)
                    if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                        best_inner = candidate
                        best_side = candidate_side
                        best_outer_angle = outer_angle
            except Exception:
                pass
    
    # Local refinement phase
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        try:
            refined = _local_refine(best_inner, best_outer_angle, n, time_limit=min(remaining * 0.6, 30.0))
            refined_side = _compute_side(refined, best_outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, best_outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Final high-precision polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            final_params = best_inner.flatten()
            res_final = minimize(_objective, final_params, args=(best_outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_final.fun < best_side:
                candidate = res_final.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, best_outer_angle)
                if _is_valid(candidate, candidate_side, best_outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Final validation
    final_side = _compute_side(best_inner, best_outer_angle)
    if not _is_valid(best_inner, final_side, best_outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
        best_outer_angle = seed_angle
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg
