"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_uv(uv_flat):
    """Objective function for 2D barycentric parameters (u, v).
    barycentric coords: (1-u-v, u, v)
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    bary0 = 1.0 - u - v
    bary1 = u
    bary2 = v
    
    # Calculate cartesian coordinates
    x = bary1 + 0.5 * bary2
    y = (np.sqrt(3) / 2) * bary2
    
    # Calculate min area
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return -min_area
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Number of restarts
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate initial points in (u, v) space
        # Uniform random in triangle
        uv_init = np.zeros((N, 2))
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            uv_init[i] = [r1, r2]
            
        x0 = uv_init.flatten()
        
        try:
            result = minimize(
                negative_min_area_uv,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 10.0:
            break
        
        uv_opt = result.x.reshape(-1, 2)
        
        # Clamp u, v to valid range [0, 1] and u+v <= 1
        valid_uv = []
        for i in range(N):
            u_i = uv_opt[i, 0]
            v_i = uv_opt[i, 1]
            
            if u_i < 0: u_i = 0.0
            if v_i < 0: v_i = 0.0
            if u_i + v_i > 1.0:
                scale = 1.0 / (u_i + v_i)
                u_i *= scale
                v_i *= scale
            
            valid_uv.append([u_i, v_i])
        
        uv_valid = np.array(valid_uv)
        
        # Convert to cartesian
        u = uv_valid[:, 0]
        v = uv_valid[:, 1]
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        points = np.column_stack((x, y))
        
        score = min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        # Fallback
        rng = np.random.RandomState(0)
        uv_init = np.zeros((N, 2))
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            uv_init[i] = [r1, r2]
        u = uv_init[:, 0]
        v = uv_init[:, 1]
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        best_points = np.column_stack((x, y))
        
    return best_points
