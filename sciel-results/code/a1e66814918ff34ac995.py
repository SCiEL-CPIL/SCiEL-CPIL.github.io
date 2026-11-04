import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _hex_vertex_offsets(angle_deg, s=1.0):
    """Vertex offsets (relative to center) for a unit hexagon at given angle."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)])

def _outer_normals(angle_deg):
    """Outward unit normals for the 6 edges of a regular hexagon (pointing outward).
    For a hexagon with vertices at angles [0,60,...,300]+angle_deg,
    the edge normals are at angles [30,90,...,330]+angle_deg."""
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + angle_deg)
    return np.column_stack([np.cos(angles), np.sin(angles)])

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
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
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
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
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
    Analytically compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    
    For a regular hexagon centered at origin with side S and rotation theta,
    the constraint for a point p is: for each of 6 outward normals n_k,
    n_k . p <= S * cos(30°) = S * sqrt(3)/2
    
    So S >= max over all inner hex vertices v and all normals n_k of (n_k . v) / (sqrt(3)/2)
    """
    n = len(inner_params)
    normals = _outer_normals(outer_angle_deg)  # shape (6, 2)
    cos30 = math.sqrt(3) / 2.0
    
    max_proj = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        offsets = _hex_vertex_offsets(a)  # shape (6, 2)
        vertices = offsets + np.array([x, y])  # shape (6, 2)
        # Project all 6 vertices onto all 6 normals: (6, 6) matrix
        projs = vertices @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / cos30

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """
    Compute smooth penalty for overlapping inner hexagons using minimum vertex-vertex distance.
    Returns 0 if all hexagons are well-separated, positive if close or overlapping.
    """
    n = len(inner_params)
    penalty = 0.0
    gap = 0.01
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            # Compute minimum distance between any pair of vertices
            min_d = float('inf')
            for vi in v1:
                for vj in v2:
                    d = np.linalg.norm(vi - vj)
                    if d < min_d:
                        min_d = d
            
            if min_d < gap:
                penalty += penalty_weight * (gap - min_d) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, outer_angle_deg, tol=1e-6):
    """Check if all inner hexagons are inside the outer hexagon and non-overlapping."""
    n = len(inner_params)
    side = _compute_side(inner_params, outer_angle_deg)
    # Check containment with a small tolerance
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            # Check against outer hexagon of the computed side
            # Use projection test
            normals = _outer_normals(outer_angle_deg)
            for nk in normals:
                if nk[0]*v[0] + nk[1]*v[1] > side * math.sqrt(3)/2 + tol:
                    return False
    # Check overlaps
    for i in range(n):
        v1 = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            sep = _poly_separation(v1, v2)
            if sep < -tol:
                return False
    return True

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    inner_params = np.array(inner)
    side = _compute_side(inner_params, 30.0) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def _coordinate_descent_polish(params, outer_angle, n, max_iters=200, step_init=0.1):
    """Coordinate descent polish: iterate over each parameter, do local 1D search."""
    params = params.copy()
    best_f = _objective(params, outer_angle, n)
    step = step_init
    
    for it in range(max_iters):
        improved = False
        for dim in range(len(params)):
            old_val = params[dim]
            # Try small perturbations
            for delta in [step, -step, step*0.5, -step*0.5, step*0.25, -step*0.25]:
                params[dim] = old_val + delta
                f = _objective(params, outer_angle, n)
                if f < best_f:
                    best_f = f
                    improved = True
                    break
            if not improved:
                params[dim] = old_val
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return params, best_f

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    safety_margin = 10.0

    n = 11
    outer_angle = 30.0

    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side

    # Build initial parameter vector from seed
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a

    # Multi-start optimization
    rng = np.random.default_rng(42)
    starts = [init_params.copy()]
    
    # Add diverse starts: perturbed seed
    for _ in range(3):
        pert = init_params.copy()
        pert[0::3] += rng.normal(0, 0.3, n)  # x
        pert[1::3] += rng.normal(0, 0.3, n)  # y
        pert[2::3] += rng.normal(0, 15, n)   # angles
        starts.append(pert)
    
    # All-zero start
    starts.append(np.zeros(3 * n))
    
    for start_params in starts:
        if time.time() > DEADLINE - safety_margin:
            break
        
        # Nelder-Mead
        try:
            res = minimize(_objective, start_params, args=(outer_angle, n), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_side:
                best_inner = []
                for i in range(n):
                    best_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
                best_side = res.fun
        except Exception:
            pass
        
        # Coordinate descent polish
        if time.time() > DEADLINE - safety_margin:
            break
        try:
            polished, pf = _coordinate_descent_polish(res.x if 'res' in dir() else start_params, 
                                                       outer_angle, n, max_iters=100, step_init=0.05)
            if pf < best_side:
                best_inner = []
                for i in range(n):
                    best_inner.append([polished[i*3], polished[i*3+1], polished[i*3+2]])
                best_side = pf
        except Exception:
            pass

    # Final validation
    if best_inner:
        inner_params = np.array(best_inner)
        side = _compute_side(inner_params, outer_angle)
        # Check validity
        if _is_valid(inner_params, outer_angle, tol=1e-5):
            best_side = side
        else:
            # Fall back to seed
            best_inner = seed_inner
            best_side = seed_side

    return best_inner, (0.0, 0.0), best_side, outer_angle
