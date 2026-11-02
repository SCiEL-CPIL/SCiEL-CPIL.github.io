import numpy as np
import time
from itertools import combinations

N = 11

def hex_to_xy(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def compute_bounding_hex(points, outer_rotation=0.0):
    pts = np.array(points)
    xs, ys = pts[:, 0], pts[:, 1]
    apothem_factor = np.sqrt(3) / 2
    
    if outer_rotation == 0:
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
    else:
        normals = np.array([
            [0.0, 1.0],
            [-np.sqrt(3)/2, 0.5],
            [-np.sqrt(3)/2, -0.5],
            [0.0, -1.0],
            [np.sqrt(3)/2, -0.5],
            [np.sqrt(3)/2, 0.5]
        ])
    
    from scipy.optimize import minimize
    cx0 = (xs.min() + xs.max()) / 2
    cy0 = (ys.min() + ys.max()) / 2
    
    def obj(c):
        cx, cy = c
        dx = xs - cx
        dy = ys - cy
        dots = np.empty((len(xs), 6))
        for j in range(6):
            dots[:, j] = normals[j, 0] * dx + normals[j, 1] * dy
        return dots.max() / apothem_factor
    
    result = minimize(obj, [cx0, cy0], method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 3000, 'adaptive': True})
    return result.fun, result.x[0], result.x[1]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q+r)) // 2
            if dist == 2:
                ring2.append((q, r))
    
    best_side = float('inf')
    best_result = None
    
    for combo in combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        hexes = base7 + [ring2[i] for i in combo]
        xy = [hex_to_xy(q, r) for q, r in hexes]
        
        for rot in [0.0, 30.0]:
            if time.time() > deadline:
                break
            side, cx, cy = compute_bounding_hex(xy, rot)
            if side < best_side:
                best_side = side
                best_result = (xy, (cx, cy), side, rot)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    xy, center, side, rot = best_result
    inner = [[x, y, 0.0] for x, y in xy]
    return inner, center, side, rot
