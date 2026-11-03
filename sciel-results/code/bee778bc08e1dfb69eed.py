"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices at angle 0
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

def _get_vertices(cx, cy, angle_deg):
    """Get vertices of a unit hexagon centered at (cx, cy) with given angle."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)],
                    [math.sin(a), math.cos(a)]])
    return rot @ _UNIT_HEX_VERTS.T + np.array([cx, cy])

def _min_dist_hexagons(v1, v2):
    """
    Compute minimum distance between two hexagons given their vertices.
    Returns negative value if they overlap.
    """
    def point_in_hex(p, hex_verts):
        x, y = p
        inside = False
        n = len(hex_verts)
        for i in range(n):
            x1, y1 = hex_verts[i]
            x2, y2 = hex_verts[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < xint:
                    inside = not inside
        return inside

    # 1. Vertex inside check
    for i in range(6):
        if point_in_hex(v1[i], v2):
            return -1.0
    for i in range(6):
        if point_in_hex(v2[i], v1):
            return -1.0

    # 2. Edge intersection check
    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1]): return True
        if d2 == 0 and min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1]): return True
        if d3 == 0 and min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1]): return True
        if d4 == 0 and min(p1[0], p2[0]) <= p4[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p4[1] <= max(p1[1], p2[1]): return True
        return False

    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j+1)%6]
            if seg_intersect(p1, p2, q1, q2):
                return -1.0

    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_d:
                min_d = d
    return min_d

def _compute_cost(params, outer_center, outer_angle):
    """
    Compute cost for a given configuration.
    params: array of shape (N, 3) -> [x, y, angle_deg]
    """
    xs = params[0::3]
    ys = params[1::3]
    angs = params[2::3]
    
    all_verts = []
    for i in range(N):
        v = _get_vertices(xs[i], ys[i], angs[i])
        all_verts.append(v)
    
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normal_angles = a + math.pi/6 + np.arange(6) * math.pi/3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for v in all_verts:
        projs = v @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    S = max_proj / (math.sqrt(3) / 2.0)
    
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = _min_dist_hexagons(all_verts[i], all_verts[j])
            if d < min_dist:
                min_dist = d
            if d < 0:
                return S + 1000.0 * (-d)
                
    return S

def _is_valid(params, outer_center, outer_angle):
    """Check if configuration is valid (no overlaps)."""
    xs = params[0::3]
    ys = params[1::3]
    angs = params[2::3]
    all_verts = []
    for i in range(N):
        v = _get_vertices(xs[i], ys[i], angs[i])
        all_verts.append(v)
    for i in range(N):
        for j in range(i+1, N):
            if _min_dist_hexagons(all_verts[i], all_verts[j]) < 0:
                return False
    return True

def _get_side_length(params, outer_center, outer_angle):
    """Get the geometric side length of the container."""
    xs = params[0::3]
    ys = params[1::3]
    angs = params[2::3]
    all_verts = []
    for i in range(N):
        v = _get_vertices(xs[i], ys[i], angs[i])
        all_verts.append(v)
    
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normal_angles = a + math.pi/6 + np.arange(6) * math.pi/3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for v in all_verts:
        projs = v @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (math.sqrt(3) / 2.0)

def construct():
    start_time = time.time()
    
    d = 2 * math.sqrt(3)
    
    init_params = []
    init_params.append([0.0, 0.0, 0.0])
    for k in range(6):
        ang = k * math.pi / 3.0
        init_params.append([d * math.cos(ang), d * math.sin(ang), 0.0])
    for k in range(5):
        ang = k * 2 * math.pi / 5.0 + math.pi/6
        r = d * math.sqrt(3)
        init_params.append([r * math.cos(ang), r * math.sin(ang), 0.0])
        
    init_flat = np.array(init_params).flatten()
    
    best_side = float('inf')
    best_params = init_flat
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    # Optimization loop
    for r in range(15):
        if time.time() > DEADLINE - 10:
            break
            
        if r == 0:
            x0 = init_flat
        else:
            x0 = init_flat + np.random.normal(0, 0.5, size=init_flat.shape)
            x0[2::3] += np.random.normal(0, 10, size=N)
            
        def objective(params):
            return _compute_cost(params, best_center, best_angle)
        
        # BFGS
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
            if _is_valid(res.x, best_center, best_angle):
                side = _get_side_length(res.x, best_center, best_angle)
                if side < best_side:
                    best_side = side
                    best_params = res.x
        except Exception:
            pass
            
        # Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if _is_valid(res.x, best_center, best_angle):
                side = _get_side_length(res.x, best_center, best_angle)
                if side < best_side:
                    best_side = side
                    best_params = res.x
        except Exception:
            pass

    # If no valid solution found, use initial (which is valid)
    if best_side == float('inf'):
        best_side = _get_side_length(init_flat, best_center, best_angle)
        best_params = init_flat

    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
