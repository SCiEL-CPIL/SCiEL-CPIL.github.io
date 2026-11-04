import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + k * np.pi / 3),
                      cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])

def edge_normals(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a + np.pi / 6 + k * np.pi / 3),
                      np.sin(a + np.pi / 6 + k * np.pi / 3)] for k in range(6)])

def required_side(points, center, angle_deg):
    normals = edge_normals(angle_deg)
    rel = points - np.array(center)
    dots = rel @ normals.T
    return np.max(dots) * 2.0 / np.sqrt(3.0)

def obj_outer(params, inner_centers):
    cx, cy, ang = params
    all_pts = np.vstack([hex_vertices(x, y, 1.0, 0.0) for x, y in inner_centers])
    return required_side(all_pts, (cx, cy), float(ang))

def make_arrangement():
    pos = [(0,0),(1,0),(2,0),
           (-1,1),(0,1),(1,1),(2,1),
           (-1,2),(0,2),(1,2),
           (0,3),(1,3)]
    return np.array([[np.sqrt(3)*(q + r/2.0), 1.5*r] for q, r in pos])

def construct():
    start = time.time()
    deadline = start + 115.0
    
    centers = make_arrangement()
    
    best_score = float('inf')
    best_params = None
    
    cx0 = np.mean(centers[:, 0])
    cy0 = np.mean(centers[:, 1])
    
    for ang in range(0, 360, 5):
        score = obj_outer((cx0, cy0, float(ang)), centers)
        if score < best_score:
            best_score = score
            best_params = (cx0, cy0, float(ang))
    
    if time.time() < deadline - 5:
        res = minimize(obj_outer, best_params, args=(centers,), method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    cx, cy, ang = best_params
    inner = [[float(x), float(y), 0.0] for x, y in centers]
    return inner, (float(cx), float(cy)), float(best_score), float(ang)
