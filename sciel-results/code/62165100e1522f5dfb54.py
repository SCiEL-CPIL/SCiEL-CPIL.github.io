import math
import time
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    SQRT3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2.0],
        [-0.5, SQRT3/2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2.0],
        [0.5, -SQRT3/2.0]
    ])

    # Outer hexagon normals
    outer_angles = [math.radians(30 + k*60) for k in range(6)]
    outer_normals = np.array([[math.cos(a), math.sin(a)] for a in outer_angles])
    
    def axial_to_cartesian(q, r):
        x = q + 0.5 * r
        y = (SQRT3 / 2.0) * r
        return x, y

    def min_side_for_cluster(points, alpha):
        cart_points = np.array([axial_to_cartesian(q, r) for q, r in points])
        
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = cart_points @ rot_mat.T
        
        verts_rot = unit_verts @ rot_mat.T
        verts_all = rotated_points[:, np.newaxis, :] + verts_rot[np.newaxis, :, :]
        verts_flat = verts_all.reshape(-1, 2)
        
        projs = verts_flat @ outer_normals.T
        max_proj = np.max(np.abs(projs))
        
        return max_proj * 2.0 / SQRT3

    def optimize_alpha(points):
        res = minimize_scalar(
            min_side_for_cluster, 
            bounds=(0, math.pi/6), 
            args=(points,), 
            method='bounded',
            options={'xatol': 1e-8}
        )
        return res.fun, res.x

    base_points = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    ring2_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0: continue
            dist_sq = q*q + q*r + r*r
            if dist_sq == 4:
                ring2_candidates.append((q, r))
                
    best_side = float('inf')
    best_points = None
    best_alpha = 0.0
    
    for combo in combinations(ring2_candidates, 5):
        if time.time() > deadline - 10:
            break
            
        points = base_points + list(combo)
        side, alpha = optimize_alpha(points)
        
        if side < best_side:
            best_side = side
            best_points = points
            best_alpha = alpha

    if best_points is None:
        pts = [(0,0)]
        for dist in range(1, 10):
            for q in range(-dist, dist+1):
                for r in range(-dist, dist+1):
                    if q*q + q*r + r*r == dist*dist:
                        if len(pts) < 12:
                            pts.append((q,r))
            if len(pts) >= 12: break
        best_points = pts[:12]
        side, alpha = optimize_alpha(best_points)
        best_side = side
        best_alpha = alpha

    inner = []
    for q, r in best_points:
        x, y = axial_to_cartesian(q, r)
        x_rot = x * math.cos(best_alpha) - y * math.sin(best_alpha)
        y_rot = x * math.sin(best_alpha) + y * math.cos(best_alpha)
        inner.append([x_rot, y_rot, math.degrees(best_alpha)])
    
    return inner, (0.0, 0.0), best_side, 0.0
