import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0 # Safety margin for 150s limit

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
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
        if _point_in_convex_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_convex_polygon(v, p1):
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

def _compute_min_side_fast(inner_params, outer_angle_deg=0.0):
    """
    Fast computation of minimum outer side using vectorized support function.
    inner_params: (N, 3) array of [x, y, angle_deg]
    """
    all_vertices = np.zeros((N * 6, 2))
    for i in range(N):
        x, y, a = inner_params[i]
        vs = _hex_vertices(x, y, a)
        all_vertices[i*6:(i+1)*6] = vs
    
    # Outer hexagon normals (flat top/bottom if angle 0? No, vertices at 0,60... => Normals at 30,90...)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for n_vec in normals:
        projs = np.dot(all_vertices, n_vec)
        max_proj = max(max_proj, np.max(projs))
    
    return 2.0 * max_proj / np.sqrt(3.0)

def _objective(params):
    """
    Objective function for optimization.
    params: flat array of length 30 (10 hexagons * 3 params).
    Hexagon 0 is fixed at (0,0, 0).
    """
    inner_params = np.zeros((N, 3))
    inner_params[0] = [0.0, 0.0, 0.0]
    
    for i in range(1, N):
        idx = (i-1)*3
        inner_params[i, 0] = params[idx]
        inner_params[i, 1] = params[idx+1]
        inner_params[i, 2] = params[idx+2]
    
    side = _compute_min_side_fast(inner_params, 0.0)
    
    # Penalty for overlap
    pen = 0.0
    for i in range(N):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, N):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.0:
                pen += 1000.0 * sep ** 2
            elif sep < 0.01:
                # Small gap penalty to encourage tight packing but allow slight separation during search
                pass 
            else:
                pass
    
    return side + pen

def construct():
    start_time = time.time()
    
    # Initial solution: Valid Flower Pattern
    # 1 center, 6 ring 1 (dist sqrt(3)), 4 ring 2 (dist 3*sqrt(3))
    # This ensures no overlap in the start.
    
    init_params = np.zeros(30)
    
    # Hex 0 is fixed at 0,0,0
    
    # Ring 1 (6 hexagons)
    r1 = np.sqrt(3)
    for k in range(6):
        angle_rad = np.deg2rad(k * 60)
        x = r1 * np.cos(angle_rad)
        y = r1 * np.sin(angle_rad)
        hex_angle = k * 60 - 30
        
        idx = k * 3
        init_params[idx] = x
        init_params[idx+1] = y
        init_params[idx+2] = hex_angle

    # Ring 2 (4 hexagons)
    r2 = 3 * np.sqrt(3)
    ring2_angles = [30, 90, 150, 210]
    for k in range(4):
        angle_rad = np.deg2rad(ring2_angles[k])
        x = r2 * np.cos(angle_rad)
        y = r2 * np.sin(angle_rad)
        hex_angle = ring2_angles[k] - 30
        
        idx = (6 + k) * 3
        init_params[idx] = x
        init_params[idx+1] = y
        init_params[idx+2] = hex_angle

    # Verify initial solution
    init_inner = np.zeros((N, 3))
    init_inner[0] = [0,0,0]
    for i in range(1, N):
        idx = (i-1)*3
        init_inner[i] = [init_params[idx], init_params[idx+1], init_params[idx+2]]
        
    init_pen = 0.0
    for i in range(N):
        v1 = _hex_vertices(init_inner[i, 0], init_inner[i, 1], init_inner[i, 2])
        for j in range(i+1, N):
            v2 = _hex_vertices(init_inner[j, 0], init_inner[j, 1], init_inner[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.0:
                init_pen += 1000.0 * sep ** 2

    if init_pen > 1e-6:
        # Fallback: If somehow invalid, spread them out more
        for i in range(1, N):
            idx = (i-1)*3
            init_params[idx] *= 1.5
            init_params[idx+1] *= 1.5

    best_params = init_params.copy()
    best_inner = init_inner.copy()
    best_side = _compute_min_side_fast(best_inner, 0.0)
    best_score = best_side # init_pen is 0 for valid start
    
    try:
        # Optimization with multiple restarts if time allows
        # We will try to improve the initial solution
        result = minimize(_objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
        
        if time.time() - start_time < DEADLINE and result.fun < best_score:
            # Verify result
            res_inner = np.zeros((N, 3))
            res_inner[0] = [0,0,0]
            for i in range(1, N):
                idx = (i-1)*3
                res_inner[i] = [result.x[idx], result.x[idx+1], result.x[idx+2]]
                
            res_pen = 0.0
            for i in range(N):
                v1 = _hex_vertices(res_inner[i, 0], res_inner[i, 1], res_inner[i, 2])
                for j in range(i+1, N):
                    v2 = _hex_vertices(res_inner[j, 0], res_inner[j, 1], res_inner[j, 2])
                    sep = _poly_separation(v1, v2)
                    if sep < 0.0:
                        res_pen += 1000.0 * sep ** 2
                        
            if res_pen < 1e-6:
                new_side = _compute_min_side_fast(res_inner, 0.0)
                if new_side < best_side:
                    best_side = new_side
                    best_params = result.x.copy()
                    best_inner = res_inner.copy()
                    
    except Exception:
        pass

    # Finalize
    inner_list = [list(row) for row in best_inner]
    
    return inner_list, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
