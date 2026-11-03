import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(x, y, angle_deg, side=1.0):
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([x + side * np.cos(angles), y + side * np.sin(angles)])

def outer_edge_normals(angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(angles), np.sin(angles)])

def required_S(all_vertices, cx, cy, outer_angle):
    normals = outer_edge_normals(outer_angle)
    max_proj = 0.0
    for verts in all_vertices:
        diff = verts - np.array([cx, cy])
        projections = diff @ normals.T
        mp = np.max(projections)
        if mp > max_proj:
            max_proj = mp
    return max_proj * 2.0 / np.sqrt(3.0)

def sat_overlap_depth(v1, v2):
    def get_normals(v):
        edges = np.roll(v, -1, axis=0) - v
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_depth = np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        if overlap <= 0:
            return 0.0
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth if min_depth < np.inf else 0.0

def objective(params):
    inner_params = params[:N*3].reshape(N, 3)
    cx, cy = params[N*3], params[N*3+1]
    outer_angle = params[N*3+2]
    
    all_vertices = []
    for i in range(N):
        x, y, ang = inner_params[i]
        verts = hex_vertices(x, y, ang)
        all_vertices.append(verts)
    
    S_req = required_S(all_vertices, cx, cy, outer_angle)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            depth = sat_overlap_depth(all_vertices[i], all_vertices[j])
            if depth > 0:
                overlap_pen += depth * depth
    
    return S_req + 100.0 * overlap_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_params = None
    best_val = np.inf
    
    inits = []
    
    # 4x3 honeycomb
    inner1 = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            inner1.append([x, y, 0.0])
    cx1, cy1 = 3.0, np.sqrt(3)
    inits.append(np.append(np.array(inner1).flatten(), [cx1, cy1, 0.0]))
    
    # 3x4 honeycomb
    inner2 = []
    for row in range(4):
        for col in range(3):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            inner2.append([x, y, 0.0])
    cx2, cy2 = 2.0, 1.5*np.sqrt(3)
    inits.append(np.append(np.array(inner2).flatten(), [cx2, cy2, 0.0]))
    
    # Circular
    inner3 = []
    for k in range(N):
        ang = 2 * np.pi * k / N
        r = 2.5
        inner3.append([r*np.cos(ang), r*np.sin(ang), np.degrees(ang)])
    inits.append(np.append(np.array(inner3).flatten(), [0.0, 0.0, 0.0]))
    
    # Random
    inner4 = []
    for k in range(N):
        inner4.append([np.random.uniform(-3,3), np.random.uniform(-3,3), np.random.uniform(0,60)])
    inits.append(np.append(np.array(inner4).flatten(), [0.0, 0.0, 0.0]))
    
    for init in inits:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except:
            pass
        
        if time.time() < deadline - 20 and best_params is not None:
            try:
                res2 = minimize(objective, best_params, method='SLSQP',
                               options={'maxiter': 100, 'ftol': 1e-12})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
            except:
                pass
    
    if best_params is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    ip = best_params[:N*3].reshape(N, 3)
    cx, cy = best_params[N*3], best_params[N*3+1]
    oa = best_params[N*3+2]
    
    av = [hex_vertices(ip[i,0], ip[i,1], ip[i,2]) for i in range(N)]
    S = required_S(av, cx, cy, oa)
    
    inner = [[float(ip[i,0]), float(ip[i,1]), float(ip[i,2])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(S), float(oa)

construct()
