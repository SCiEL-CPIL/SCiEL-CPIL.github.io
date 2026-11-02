import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
SQRT3 = np.sqrt(3.0)

def hex_vertices(center, side, angle):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9:
            return False
    return True

def polygons_intersect(poly1, poly2):
    def project_polygon(axis, poly):
        projections = np.dot(poly, axis)
        return np.min(projections), np.max(projections)

    def get_edges(poly):
        edges = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            edges.append(normal)
        return edges

    axes = get_edges(poly1) + get_edges(poly2)
    
    for axis in axes:
        min1, max1 = project_polygon(axis, poly1)
        min2, max2 = project_polygon(axis, poly2)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
    if outer_side <= 0:
        return False, np.inf

    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    n_inner = len(inner)
    inner_verts = []
    
    for i in range(n_inner):
        x, y, ang = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(ang))
        inner_verts.append(v)
        
    for i in range(n_inner):
        v = inner_verts[i]
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False, np.inf

    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                return False, np.inf
                
    return True, 0.0

def min_distance_to_polygon_boundary(point, polygon):
    min_d = np.inf
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        d = distance_point_to_segment(point, p1, p2)
        if d < min_d:
            min_d = d
    return min_d

def distance_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_distance_between_polygons(poly1, poly2):
    if polygons_intersect(poly1, poly2):
        min_d = np.inf
        for v in poly1:
            d = min_distance_to_polygon_boundary(v, poly2)
            if d < min_d:
                min_d = d
        return -min_d * 10.0
    
    min_d = np.inf
    for v in poly1:
        d = min_distance_to_polygon_boundary(v, poly2)
        if d < min_d:
            min_d = d
    for v in poly2:
        d = min_distance_to_polygon_boundary(v, poly1)
        if d < min_d:
            min_d = d
    return min_d

def penalty_function(params):
    outer_side = params[0]
    outer_cx, outer_cy = params[1:3]
    outer_angle = params[3]
    
    inner_params = params[4:]
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    inner_verts = []
    for i in range(N):
        x, y, a = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(a))
        inner_verts.append(v)
        
    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    
    total_penalty = 0.0
    
    for v in inner_verts:
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                dist = min_distance_to_polygon_boundary(pt, outer_verts)
                total_penalty += dist * dist * 100.0

    for i in range(N):
        for j in range(i + 1, N):
            min_dist = min_distance_between_polygons(inner_verts[i], inner_verts[j])
            if min_dist < 0:
                total_penalty += (-min_dist) * 100.0
            elif min_dist < 1e-3:
                total_penalty += (1e-3 - min_dist) * 10.0

    return outer_side + total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = np.inf
    best_params = None
    
    def init_grid():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        for i in range(N):
            col = i % 4
            row = i // 4
            ix = col * 3.5
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    def init_hex_cluster():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        
        for k in range(6):
            idx = k + 1
            ang = np.radians(k * 60)
            dist = 2.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        for k in range(5):
            idx = k + 7
            ang = np.radians(k * 72 + 30)
            dist = 4.0
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            
        return params

    initializations = [init_grid(), init_hex_cluster()]
    
    for init_params in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(penalty_function, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    if best_params is None:
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        return inner, (0, 0), 10.0, 0.0

    outer_side = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 3*i]
        y = best_params[4 + 3*i+1]
        a = best_params[4 + 3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    valid, _ = check_validity(inner, outer_center, outer_side, outer_angle)
    
    if not valid:
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        outer_side = 10.0
        outer_center = (0, 0)
        outer_angle = 0.0
        
    return inner, outer_center, outer_side, outer_angle
