import numpy as np
from scipy.optimize import minimize
import time

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def hex_edge_normals(v):
    normals = []
    center = v.mean(axis=0)
    for i in range(6):
        p1 = v[i]
        p2 = v[(i+1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        normals.append(n)
    return np.array(normals)

def are_separated(v1, v2, eps=1e-6):
    n1 = hex_edge_normals(v1)
    n2 = hex_edge_normals(v2)
    all_n = np.vstack([n1, n2])
    for n in all_n:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if max(proj1) < min(proj2) - eps or max(proj2) < min(proj1) - eps:
            return True
    return False

def min_enclosing_hexagon_size(points, angle_deg=0):
    """Find minimum side length of regular hexagon with given angle that contains all points."""
    a = np.radians(angle_deg)
    # 6 outward normals
    normals = []
    for i in range(6):
        ang = a + np.radians(60*i + 30)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    # For each normal, the constraint is: max over points of (p-c)·n <= s*sqrt(3)/2
    # To minimize s, we need to find c that minimizes max over n of max over p of (p-c)·n
    # This is equivalent to finding the smallest hexagon
    
    # Simple approach: try center at centroid
    c = points.mean(axis=0)
    max_proj = 0
    for n in normals:
        proj = (points - c) @ n
        max_proj = max(max_proj, proj.max())
    s = 2 * max_proj / np.sqrt(3)
    return s

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580
    
    s3 = np.sqrt(3)
    inner0 = np.array([
        [0, 0],
        [s3, 0],
        [-s3, 0],
        [s3/2, 1.5],
        [-s3/2, 1.5],
        [s3/2, -1.5],
        [-s3/2, -1.5],
        [2*s3, 0],
        [s3, 3],
        [-s3, 3],
        [0, -3],
    ])
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(11, 2)
        hexes = [hex_vertices(pos[i,0], pos[i,1], 1.0, 0) for i in range(11)]
        
        # Check separation
        pen = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not are_separated(hexes[i], hexes[j]):
                    n1 = hex_edge_normals(hexes[i])
                    n2 = hex_edge_normals(hexes[j])
                    all_n = np.vstack([n1, n2])
                    min_gap = np.inf
                    for n in all_n:
                        proj1 = hexes[i] @ n
                        proj2 = hexes[j] @ n
                        gap = max(min(proj2) - max(proj1), min(proj1) - max(proj2))
                        min_gap = min(min_gap, gap)
                    if min_gap < 0:
                        pen += (-min_gap)**2
        
        if pen > 0:
            return 1e10 + pen
        
        # Compute minimal enclosing hexagon
        all_points = np.vstack(hexes)
        s = min_enclosing_hexagon_size(all_points, 0)
        return s
    
    x0 = inner0.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
    
    if time.time() > deadline - 5:
        # Return initial
        inner = np.column_stack([inner0, np.zeros(11)])
        outer = np.array([0.0, 0.0, 0.0])
        all_points = np.vstack([hex_vertices(inner0[i,0], inner0[i,1], 1.0, 0) for i in range(11)])
        side = min_enclosing_hexagon_size(all_points, 0)
        return inner, outer, side
    
    pos_opt = result.x.reshape(11, 2)
    hexes = [hex_vertices(pos_opt[i,0], pos_opt[i,1], 1.0, 0) for i in range(11)]
    
    # Verify separation
    for i in range(11):
        for j in range(i+1, 11):
            if not are_separated(hexes[i], hexes[j]):
                inner = np.column_stack([inner0, np.zeros(11)])
                outer = np.array([0.0, 0.0, 0.0])
                all_points = np.vstack([hex_vertices(inner0[i,0], inner0[i,1], 1.0, 0) for i in range(11)])
                side = min_enclosing_hexagon_size(all_points, 0)
                return inner, outer, side
    
    all_points = np.vstack(hexes)
    side = min_enclosing_hexagon_size(all_points, 0)
    
    inner = np.column_stack([pos_opt, np.zeros(11)])
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, side
