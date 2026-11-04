import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon (flat-top at angle 0), side s, centered at (cx,cy)."""
    a = math.radians(angle_deg)
    v = np.empty((6, 2))
    for i in range(6):
        t = a + i * math.pi / 3.0
        v[i] = [cx + s * math.cos(t), cy + s * math.sin(t)]
    return v

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and angle angle_deg."""
    dx = px - cx
    dy = py - cy
    a = math.radians(-angle_deg)
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    
    h = s * math.sqrt(3) / 2
    if rx > s or rx < -s: return False
    if ry > h or ry < -h: return False
    if math.sqrt(3)*rx + ry > s*math.sqrt(3): return False
    if -math.sqrt(3)*rx + ry > s*math.sqrt(3): return False
    if -math.sqrt(3)*rx - ry > s*math.sqrt(3): return False
    if math.sqrt(3)*rx - ry > s*math.sqrt(3): return False
    return True

def enclosing_hex_penalty(points, cx, cy, s, angle_deg):
    """Penalty if any point is outside the hexagon."""
    pen = 0.0
    for p in points:
        if not point_in_hex(p[0], p[1], cx, cy, s, angle_deg):
            dx = p[0] - cx
            dy = p[1] - cy
            a = math.radians(-angle_deg)
            rx = dx * math.cos(a) - dy * math.sin(a)
            ry = dx * math.sin(a) + dy * math.cos(a)
            h = s * math.sqrt(3) / 2
            v1 = max(0, rx - s)
            v2 = max(0, -rx - s)
            v3 = max(0, ry - h)
            v4 = max(0, -ry - h)
            v5 = max(0, math.sqrt(3)*rx + ry - s*math.sqrt(3))
            v6 = max(0, -math.sqrt(3)*rx + ry - s*math.sqrt(3))
            pen += v1**2 + v2**2 + v3**2 + v4**2 + v5**2 + v6**2
    return pen

def get_grid_positions(topology, gap_factor, rotation_deg):
    """
    Generate centers for 12 unit hexagons on a grid.
    gap_factor >= 1.0 ensures non-overlap (1.0 is touching).
    """
    centers = []
    # Base lattice vectors for hexagonal packing (touching)
    # dx = 1.5 * side, dy = sqrt(3) * side
    # For unit hexagons, distance between centers is sqrt(3) for adjacent?
    # No, for flat-top hexagons side 1:
    # Width (x) = 2, Height (y) = sqrt(3).
    # Horizontal spacing: 1.5 (3/4 of width? No, 3/2 of side? No.)
    # Let's use axial coordinates converted to cartesian.
    # q, r -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    # This gives touching hexagons.
    
    if topology == '2x6':
        # 2 rows, 6 cols
        for i in range(6):
            for j in range(2):
                q = i
                r = j
                x = 1.5 * q
                y = math.sqrt(3) * (r + q / 2.0)
                centers.append((x, y))
    elif topology == '3x4':
        for i in range(4):
            for j in range(3):
                q = i
                r = j
                x = 1.5 * q
                y = math.sqrt(3) * (r + q / 2.0)
                centers.append((x, y))
    elif topology == 'hex_cluster':
        # Center (0,0)
        centers.append((0, 0))
        # Ring 1 (6 hexes)
        for i in range(6):
            q = [1, 0, -1, -1, 0, 1][i]
            r = [0, 1, 1, 0, -1, -1][i]
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            centers.append((x, y))
        # Ring 2 partial (5 hexes) to make 12
        # Add (2,0), (1,1), (0,2), (-1,2), (-2,1)
        extra = [(2,0), (1,1), (0,2), (-1,2), (-2,1)]
        for q, r in extra:
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            centers.append((x, y))
    else:
        # Fallback line
        for i in range(12):
            centers.append((1.5 * i, 0))
            
    # Apply gap factor (scale coordinates from centroid)
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    
    scaled_centers = []
    for x, y in centers:
        # Scale relative to centroid to increase gaps
        sx = (x - cx_mean) * gap_factor + cx_mean
        sy = (y - cy_mean) * gap_factor + cy_mean
        scaled_centers.append((sx, sy))
        
    # Apply rotation
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    rotated_centers = []
    for x, y in scaled_centers:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        rotated_centers.append((rx, ry))
        
    return rotated_centers

def optimize_configuration(topology):
    """Optimize gap, rotation, and outer hexagon for a given topology."""
    
    def objective(params):
        gap_log = params[0]
        gap_factor = 1.0 + math.exp(gap_log) # Ensure gap > 1.0
        rot = params[1]
        ox = params[2]
        oy = params[3]
        os = params[4]
        oa = params[5]
        
        if os < 0.1:
            os = 0.1
            
        centers = get_grid_positions(topology, gap_factor, rot)
        
        # Generate vertices for all inner hexagons (angle 0)
        points = []
        for cx, cy in centers:
            v = hex_vertices(cx, cy, 1.0, 0.0)
            points.extend(v.tolist())
            
        pen = enclosing_hex_penalty(points, ox, oy, os, oa)
        return os + 100.0 * pen

    # Initial guess
    # Get centers for gap=1.1
    centers_init = get_grid_positions(topology, 1.1, 0.0)
    cx_init = np.mean([c[0] for c in centers_init])
    cy_init = np.mean([c[1] for c in centers_init])
    
    # Estimate outer size
    xs = [c[0] for c in centers_init]
    ys = [c[1] for c in centers_init]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    s_init = max(w, h) / 1.5 + 2.0
    
    x0 = [math.log(0.1), 0.0, cx_init, cy_init, s_init, 0.0]
    
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
    
    gap_factor = 1.0 + math.exp(res.x[0])
    rot = res.x[1]
    centers = get_grid_positions(topology, gap_factor, rot)
    
    # Final check and refinement of outer hex
    points = []
    for cx, cy in centers:
        v = hex_vertices(cx, cy, 1.0, 0.0)
        points.extend(v.tolist())
        
    # Re-optimize outer hex only for precision
    def outer_obj(p):
        ox, oy, os, oa = p
        if os < 0.1: os = 0.1
        return os + 100.0 * enclosing_hex_penalty(points, ox, oy, os, oa)
        
    x0_outer = [res.x[2], res.x[3], res.x[4], res.x[5]]
    res_outer = minimize(outer_obj, x0_outer, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
    
    ox, oy, os, oa = res_outer.x
    pen = enclosing_hex_penalty(points, ox, oy, os, oa)
    
    if pen < 1e-4:
        return [[cx, cy, 0.0] for cx, cy in centers], (ox, oy, os, oa), os
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = float('inf')
    best_inner = None
    best_outer = None
    
    topologies = ['2x6', '3x4', 'hex_cluster']
    
    for top in topologies:
        if time.time() > deadline:
            break
            
        result = optimize_configuration(top)
        if result:
            inner, outer, score = result
            if score < best_score:
                best_score = score
                best_inner = inner
                best_outer = outer
                
    if best_inner is None:
        # Fallback
        centers = get_grid_positions('2x6', 1.5, 0.0)
        points = []
        for cx, cy in centers:
            v = hex_vertices(cx, cy, 1.0, 0.0)
            points.extend(v.tolist())
        cx0 = np.mean([p[0] for p in points])
        cy0 = np.mean([p[1] for p in points])
        s0 = 15.0
        best_inner = [[cx, cy, 0.0] for cx, cy in centers]
        best_outer = (cx0, cy0, s0, 0.0)
        best_score = s0

    return best_inner, (float(best_outer[0]), float(best_outer[1])), float(best_outer[2]), float(best_outer[3])
