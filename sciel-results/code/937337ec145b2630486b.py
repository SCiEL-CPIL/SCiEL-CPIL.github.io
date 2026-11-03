import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 110.0

def hex_vertices(cx, cy, r, angle_deg):
    angles = np.deg2rad(angle_deg + np.arange(0, 360, 60))
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def point_in_hexagon(p, verts):
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        v = p2 - p1
        w = p - p1
        cross = v[0] * w[1] - v[1] * w[0]
        if cross < -1e-9:
            return False
    return True

def point_to_segment_dist(p, v1, v2):
    v = v2 - v1
    w = p - v1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - v1)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - v2)
    b = c1 / c2
    v_proj = v1 + b * v
    return np.linalg.norm(p - v_proj)

def min_dist_point_to_polygon(p, verts):
    min_d = np.inf
    n = len(verts)
    for i in range(n):
        d = point_to_segment_dist(p, verts[i], verts[(i+1)%n])
        if d < min_d:
            min_d = d
    return min_d

def polygons_overlap_sat(v1, v2):
    def project(poly, axis):
        proj = np.dot(poly, axis)
        return np.min(proj), np.max(proj)
    
    n1 = len(v1)
    n2 = len(v2)
    
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        axis = np.array([-edge[1], edge[0]])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9: continue
        axis = axis / axis_norm
        
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        
        if max1 < min2 or max2 < min1:
            return False
        
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        axis = np.array([-edge[1], edge[0]])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9: continue
        axis = axis / axis_norm
        
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def objective(params):
    start_time = time.time()
    if start_time > DEADLINE:
        return 1e9
        
    inner_params = params[:36].reshape(12, 3)
    L = params[36]
    outer_angle = params[37]
    
    if L < 0.5: # Minimum possible size
        return 1e9
        
    outer_verts = hex_vertices(0, 0, L, outer_angle)
    
    penalty = 0.0
    inner_verts_list = []
    
    for i in range(12):
        x, y, ang = inner_params[i]
        verts = hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        
        # Check containment
        for v in verts:
            if not point_in_hexagon(v, outer_verts):
                d = min_dist_point_to_polygon(v, outer_verts)
                penalty += d * 10.0 

    # Check overlaps
    for i in range(12):
        for j in range(i + 1, 12):
            if polygons_overlap_sat(inner_verts_list[i], inner_verts_list[j]):
                # Calculate a rough penetration penalty
                # Sum of distances of vertices of i to boundary of j (if inside)
                pen = 0.0
                for v in inner_verts_list[i]:
                    if point_in_hexagon(v, inner_verts_list[j]):
                        pen += min_dist_point_to_polygon(v, inner_verts_list[j])
                for v in inner_verts_list[j]:
                    if point_in_hexagon(v, inner_verts_list[i]):
                        pen += min_dist_point_to_polygon(v, inner_verts_list[i])
                penalty += (pen + 1.0) * 10.0

    return L + penalty

def construct():
    start_time = time.time()
    
    # Initialization: Honeycomb
    points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = 1.5 * q * np.sqrt(3)
            y = np.sqrt(3) * (r + q / 2.0) * np.sqrt(3)
            dist = np.sqrt(x**2 + y**2)
            points.append((dist, x, y))
            
    points.sort()
    init_inner = []
    for i in range(12):
        _, x, y = points[i]
        init_inner.append([x, y, 0.0])
        
    init_params = np.array(init_inner).flatten()
    max_r = np.max(np.sqrt(init_inner[:, 0]**2 + init_inner[:, 1]**2))
    init_L = max_r + 1.5
    init_outer_angle = 0.0
    
    init_params = np.append(init_params, [init_L, init_outer_angle])
    
    best_params = init_params.copy()
    best_cost = objective(init_params)
    
    options = {
        'maxiter': 3000,
        'xatol': 1e-5,
        'fatol': 1e-5,
        'disp': False
    }
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', options=options)
        if time.time() < DEADLINE:
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
    except Exception:
        pass
        
    if time.time() > DEADLINE:
        return best_params[:36].reshape(12, 3), (0.0, 0.0), best_params[36], best_params[37]
        
    inner = best_params[:36].reshape(12, 3)
    outer_side = best_params[36]
    outer_angle = best_params[37]
    
    return inner, (0.0, 0.0), outer_side, outer_angle
