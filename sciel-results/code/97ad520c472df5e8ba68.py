import numpy as np
import scipy.optimize as opt
import time
import math

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0 + angle_rad
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def polygon_overlap_depth(v1, v2):
    min_depth = float('inf')
    
    def get_edges(vertices):
        edges = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i+1) % len(vertices)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal = normal / norm
            edges.append(normal)
        return edges
    
    normals = get_edges(v1) + get_edges(v2)
    
    for n in normals:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    if min_depth < 1e-7:
        return 0.0
    return min_depth

def min_side_length_for_containment(all_vertices):
    if len(all_vertices) == 0:
        return 0.0
    pts = np.vstack(all_vertices)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    proj = np.dot(pts, normals.T)
    max_abs_proj = np.max(np.abs(proj), axis=0)
    required_apothem = np.max(max_abs_proj)
    S = required_apothem * 2.0 / math.sqrt(3.0)
    return S

def objective(params):
    positions = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    vertices_list = []
    for i in range(N):
        v = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
        vertices_list.append(v)
    
    S_min = min_side_length_for_containment(vertices_list)
    
    total_overlap = 0.0
    for i in range(N):
        for j in range(i+1, N):
            depth = polygon_overlap_depth(vertices_list[i], vertices_list[j])
            if depth > 0:
                total_overlap += depth
    
    penalty_factor = 100.0
    return S_min + penalty_factor * total_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_params = None
    
    d = math.sqrt(3.0)
    
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * d + j * (d / 2.0)
            y = j * (d * math.sqrt(3.0) / 2.0)
            dist = math.sqrt(x*x + y*y)
            lattice_points.append((dist, x, y))
            
    lattice_points.sort()
    init_positions = [(x, y) for _, x, y in lattice_points[:N]]
    
    init_params = []
    for i in range(N):
        init_params.extend([init_positions[i][0], init_positions[i][1], 0.0])
        
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            params = np.array(init_params)
        else:
            params = np.array(init_params) + np.random.normal(0, 0.5, size=3*N)
            
        try:
            res = opt.minimize(objective, params, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 5000})
            
            # Check if this result is valid (no overlap)
            # Re-calculate overlap for res.x
            pos = []
            ang = []
            for i in range(N):
                pos.append((res.x[3*i], res.x[3*i+1]))
                ang.append(res.x[3*i+2])
                
            verts = [get_hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
            
            is_valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if polygon_overlap_depth(verts[i], verts[j]) > 1e-5:
                        is_valid = False
                        break
                if not is_valid:
                    break
                    
            if is_valid:
                S_val = min_side_length_for_containment(verts)
                if S_val < best_score:
                    best_score = S_val
                    best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        verts = [get_hex_vertices(init_positions[i][0], init_positions[i][1], 0.0) for i in range(N)]
        S = min_side_length_for_containment(verts)
        inner = [[init_positions[i][0], init_positions[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), S, 0.0
        
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        
    verts = [get_hex_vertices(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)]
    final_S = min_side_length_for_containment(verts)
    
    return inner, (0.0, 0.0), final_S, 0.0
