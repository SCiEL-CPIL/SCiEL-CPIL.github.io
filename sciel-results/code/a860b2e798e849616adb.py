# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def get_hex_vertices_fast(cx, cy, a_deg):
    # Precomputed cos/sin for unit hexagon
    # Angles: a, a+60, ...
    a_rad = np.deg2rad(a_deg)
    # cos(a + 60k) = cos(a)cos(60k) - sin(a)sin(60k)
    # sin(a + 60k) = sin(a)cos(60k) + cos(a)sin(60k)
    ca = np.cos(a_rad)
    sa = np.sin(a_rad)
    
    # cos(0)=1, sin(0)=0
    # cos(60)=0.5, sin(60)=sqrt(3)/2
    # cos(120)=-0.5, sin(120)=sqrt(3)/2
    # cos(180)=-1, sin(180)=0
    # cos(240)=-0.5, sin(240)=-sqrt(3)/2
    # cos(300)=0.5, sin(300)=-sqrt(3)/2
    
    c_vals = np.array([1, 0.5, -0.5, -1, -0.5, 0.5])
    s_vals = np.array([0, np.sqrt(3)/2, np.sqrt(3)/2, 0, -np.sqrt(3)/2, -np.sqrt(3)/2])
    
    vx = ca * c_vals - sa * s_vals
    vy = sa * c_vals + ca * s_vals
    
    return np.stack([cx + vx, cy + vy], axis=1)

def point_in_hex(pt, hex_verts):
    # pt: (2,)
    # hex_verts: (6,2)
    # Returns True if pt is inside or on boundary
    # Use cross product method
    x, y = pt
    n = 6
    p1x, p1y = hex_verts[0]
    for i in range(n):
        p2x, p2y = hex_verts[(i + 1) % n]
        # Check if point is on the "left" of edge p1->p2 (assuming CCW)
        # Cross product (p2-p1) x (p-pt)
        # If all are >= 0 (or <= 0), it's inside.
        # Hexagons are convex.
        # Let's check orientation.
        # Vertices are generated CCW.
        # So cross product should be >= 0 for all edges if inside.
        cross = (p2x - p1x) * (y - p1y) - (p2y - p1y) * (x - p1x)
        if cross < -1e-9: # Small tolerance
            return False
        p1x, p1y = p2x, p2y
    return True

def check_overlap(hex1, hex2):
    # hex1, hex2 are (6,2) arrays
    # Check if any vertex of hex1 is in hex2
    for v in hex1:
        if point_in_hex(v, hex2):
            return True
    # Check if any vertex of hex2 is in hex1
    for v in hex2:
        if point_in_hex(v, hex1):
            return True
    return False

def calculate_min_side(hex_data):
    # hex_data: (11, 3)
    # Outer hex normals: 30, 90, ...
    normals = []
    for k in range(6):
        angle = np.deg2rad(30 + 60 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    max_proj = 0
    for i in range(11):
        x, y, a = hex_data[i]
        verts = get_hex_vertices_fast(x, y, a)
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (np.sqrt(3) / 2)

def objective(params):
    hex_data = params.reshape(11, 3)
    
    # Check overlaps
    verts_list = []
    for i in range(11):
        x, y, a = hex_data[i]
        verts_list.append(get_hex_vertices_fast(x, y, a))
        
    for i in range(11):
        for j in range(i + 1, 11):
            if check_overlap(verts_list[i], verts_list[j]):
                return 1e6
    
    return calculate_min_side(hex_data)

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600 - 10
    
    s3 = np.sqrt(3)
    init = [
        [0, 0, 0],
        [s3, 0, 0],
        [-s3, 0, 0],
        [s3/2, 1.5, 0],
        [-s3/2, 1.5, 0],
        [s3/2, -1.5, 0],
        [-s3/2, -1.5, 0],
        [3*s3/2, 1.5, 0],
        [3*s3/2, -1.5, 0],
        [-3*s3/2, 1.5, 0],
        [-3*s3/2, -1.5, 0]
    ]
    
    best_params = np.array(init).flatten()
    best_side = calculate_min_side(best_params.reshape(11,3))
    
    # Optimization
    try:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if res.fun < best_side:
            best_params = res.x
            best_side = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline:
        # Try to polish with tighter tolerance
        try:
             res2 = minimize(objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
             if res2.fun < best_side:
                 best_params = res2.x
                 best_side = res2.fun
        except Exception:
             pass

    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
