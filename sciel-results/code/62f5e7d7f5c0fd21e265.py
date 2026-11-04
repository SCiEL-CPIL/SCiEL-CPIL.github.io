import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0

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
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap, positive if separated.
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

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """
    Compute penalty for overlapping inner hexagons.
    inner_params: array of shape (n, 3)
    """
    n = len(inner_params)
    penalty = 0.0
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
            
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                # Quadratic penalty for violation
                penalty += penalty_weight * (0.01 - sep) ** 2
    return penalty

def _compute_min_side(inner_params, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (centered at 0,0)
    to contain all inner hexagons.
    """
    n = len(inner_params)
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        vs = _hex_vertices(x, y, a)
        all_vertices.extend(vs)
    
    all_vertices = np.array(all_vertices)
    
    # Outer hexagon normals
    # Vertices of unit hexagon (side 1) at angle 0:
    # The support function h(n) = max dot(v, n)
    # For a regular hexagon of side S, centered at 0, angle alpha:
    # The distance from center to side (apothem) is S * sqrt(3)/2.
    # The normals to the sides are at angles alpha + 30 + k*60 ?
    # Let's check:
    # Vertices at 0, 60, ...
    # Midpoints of sides (apothem directions) at 30, 90, ...
    # So normals are at alpha + 30 + k*60.
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for n_vec in normals:
        projs = np.dot(all_vertices, n_vec)
        max_proj = max(max_proj, np.max(projs))
        
    # S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def _objective(params):
    """
    Objective function for optimizer.
    params: [x1, y1, a1, ..., x11, y11, a11, outer_angle]
    Note: We fix the first hexagon (center) at (0,0) and angle 30 to reduce DOF.
    So params contains 10 * 3 + 1 = 31 variables.
    """
    # Unpack
    outer_angle = params[-1]
    # Reconstruct full inner params
    inner_params = np.zeros((N, 3))
    inner_params[0] = [0.0, 0.0, 30.0] # Fixed center
    for i in range(1, N):
        idx = (i-1)*3
        inner_params[i, 0] = params[idx]
        inner_params[i, 1] = params[idx+1]
        inner_params[i, 2] = params[idx+2]
        
    side = _compute_min_side(inner_params, outer_angle)
    penalty = _overlap_penalty(inner_params)
    
    return side + penalty

def _seed_solution():
    """Return the deterministic flower packing."""
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
    
    # Compute side
    params = np.zeros(31)
    for i in range(1, N):
        idx = (i-1)*3
        params[idx] = inner[i][0]
        params[idx+1] = inner[i][1]
        params[idx+2] = inner[i][2]
    params[-1] = 30.0 # Outer angle
    
    side = _compute_min_side(np.array(inner), 30.0)
    return inner, (0.0, 0.0), side, 30.0, params

def construct():
    start_time = time.time()
    
    # Get seed
    seed_inner, seed_center, seed_side, seed_angle, seed_params = _seed_solution()
    
    best_side = seed_side
    best_params = seed_params.copy()
    
    # Optimization
    # Use Nelder-Mead for robustness
    # Bounds: x, y in [-5, 5], angle in [0, 60]
    # We don't enforce bounds strictly in Nelder-Mead, but we can wrap the objective
    # to penalize out-of-bounds or just let it run. The seed is good, so it should stay close.
    
    try:
        # Run optimization
        result = minimize(_objective, seed_params, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        
        if result.fun < best_side:
            best_params = result.x
            # Recalculate side without penalty for reporting
            outer_angle = best_params[-1]
            inner_params = np.zeros((N, 3))
            inner_params[0] = [0.0, 0.0, 30.0]
            for i in range(1, N):
                idx = (i-1)*3
                inner_params[i, 0] = best_params[idx]
                inner_params[i, 1] = best_params[idx+1]
                inner_params[i, 2] = best_params[idx+2]
            
            # Check validity (no overlap)
            penalty = _overlap_penalty(inner_params)
            if penalty < 1e-5:
                best_side = _compute_min_side(inner_params, outer_angle)
                # Update best_inner
                seed_inner = []
                for i in range(N):
                    seed_inner.append([inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]])
                seed_angle = outer_angle
    except Exception as e:
        # If optimization fails, keep seed
        pass
    
    # Final check and formatting
    # Ensure strict disjointness by adding a tiny margin to side
    final_side = best_side + 1e-9
    
    return seed_inner, seed_center, final_side, seed_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
