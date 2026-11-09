"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy import optimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    angles = np.radians(angles)
    x = side * np.cos(angles) + cx
    y = side * np.sin(angles) + cy
    return np.column_stack((x, y))

def _solve_enclosing_lp(points):
    angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    A_ub = []
    b_ub = []
    
    apothem = np.sqrt(3) / 2
    
    for p in points:
        for n in normals:
            A_ub.append([-n[0], -n[1], -apothem])
            b_ub.append(-(p[0] * n[0] + p[1] * n[1]))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
    
    try:
        res = optimize.linprog(c=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            cx, cy, S = res.x
            return S, cx, cy
        else:
            return None
    except Exception:
        return None

def _triangular_config(n):
    centers = []
    d = np.sqrt(3) + 0.01
    rows = [4, 4, 4]
    for i, k in enumerate(rows):
        for j in range(k):
            x = j * d + (i % 2) * d / 2
            y = i * d * np.sqrt(3) / 2
            centers.append([x, y])
    
    if centers:
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        centers = [[c[0] - cx, c[1] - cy] for c in centers]
    
    return centers

def _line_config(n):
    d = 2.0 + 0.01
    centers = []
    for i in range(n):
        centers.append([i * d, 0])
    cx = np.mean([c[0] for c in centers])
    cy = 0
    centers = [[c[0] - cx, c[1] - cy] for c in centers]
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    best_S = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    configs = []
    configs.append(_triangular_config(N))
    configs.append(_line_config(N))
    
    for cfg in configs:
        all_points = []
        for c in cfg:
            verts = _hex_vertices(c[0], c[1], 1.0, 0.0)
            all_points.extend(verts.tolist())
        
        all_points = np.array(all_points)
        
        res = _solve_enclosing_lp(all_points)
        if res:
            S, cx, cy = res
            if S < best_S:
                best_S = S
                best_inner = [[c[0], c[1], 0.0] for c in cfg]
                best_center = (cx, cy)
                best_angle = 0.0
        
        if time.time() > deadline:
            break
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
