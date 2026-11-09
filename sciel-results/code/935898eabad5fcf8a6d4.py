# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    xs = cx + d * np.cos(angles)
    ys = cy + d * np.sin(angles)
    return np.column_stack([xs, ys])

def _min_outer_side(centers, inner_angle_deg, outer_angle_deg):
    """Min side length of outer hexagon at origin with given angle containing all inner unit hexagons."""
    oa = np.radians(outer_angle_deg)
    normal_angles = oa + np.radians(30) + np.radians(60) * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    ia = np.radians(inner_angle_deg)
    for cx, cy in centers:
        verts = _hex_vertices(cx, cy, 1.0, np.degrees(ia))
        for j in range(6):
            proj = verts @ normals[j]
            max_proj = max(max_proj, np.max(proj))
    
    return 2.0 * max_proj / np.sqrt(3.0)

def hexagon_packing_12():
    t0 = time.time()
    deadline = t0 + 590.0
    
    arrangements = [
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1),(2,0),(-2,0),(0,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1),(2,1),(-2,-1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(-1,2),(0,2),(1,2)],
    ]
    
    best_score = -np.inf
    best_data = None
    
    for arr_idx, pts in enumerate(arrangements):
        if time.time() > deadline:
            break
        pts = np.array(pts, dtype=float)
        
        def obj(p):
            d, rot, orot = p
            if d < np.sqrt(3) - 1e-9:
                return 1e10
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            e1 = d * np.array([1.0, 0.0])
            e2 = d * np.array([0.5, np.sqrt(3)/2.0])
            centers = (pts @ np.array([e1, e2]).T) @ R.T
            s = _min_outer_side(centers, np.degrees(rot), np.degrees(orot))
            if not np.isfinite(s) or s < np.sqrt(12) - 1e-9:
                return 1e10
            return -1.0/s
        
        best_here = np.inf
        best_x = None
        for orot_init in [0.0, np.radians(30)]:
            if time.time() > deadline:
                break
            x0 = [np.sqrt(3), 0.0, orot_init]
            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter':3000,'xatol':1e-7,'fatol':1e-9})
                if res.fun < best_here:
                    best_here = res.fun
                    best_x = res.x
            except:
                pass
        
        if best_x is not None and best_here < 1e9:
            d, rot, orot = best_x
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            e1 = d * np.array([1.0, 0.0])
            e2 = d * np.array([0.5, np.sqrt(3)/2.0])
            centers = (pts @ np.array([e1, e2]).T) @ R.T
            s = _min_outer_side(centers, np.degrees(rot), np.degrees(orot))
            if s >= np.sqrt(12) - 1e-9:
                score = 1.0/s
                if score > best_score:
                    best_score = score
                    best_data = (centers, rot, orot, s)
    
    if best_data is None:
        inner = np.array([[0,0,0],[-2.5,0,0],[2.5,0,0],[-1.25,2.17,0],[1.25,2.17,0],
                         [-1.25,-2.17,0],[1.25,-2.17,0],[-3.75,2.17,0],[3.75,2.17,0],
                         [-3.75,-2.17,0],[3.75,-2.17,0],[0,-4,0]], dtype=float)
        return inner, np.array([0.0,0.0,0.0]), 8.0
    
    centers, rot, orot, s = best_data
    inner = np.column_stack([centers, np.full(len(centers), np.degrees(rot))])
    outer = np.array([0.0, 0.0, np.degrees(orot)])
    return inner, outer, s

# EVOLVE-BLOCK-END
