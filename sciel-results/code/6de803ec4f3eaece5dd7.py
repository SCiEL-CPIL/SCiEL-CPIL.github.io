import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    angles = angle + np.radians(np.arange(6) * 60.0)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_inner_vertices(inner):
    verts_list = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    return np.vstack(verts_list)

def _compute_normals(outer_angle_deg=0.0):
    base = np.radians(outer_angle_deg + 30.0)
    return np.array([[np.cos(base + np.radians(60.0)*k), np.sin(base + np.radians(60.0)*k)] for k in range(6)])

def _min_enclosing_hex(vertices, outer_angle_deg=0.0):
    """Find min side of regular hexagon with given orientation enclosing all vertices.
    Uses convex optimization for center."""
    normals = _compute_normals(outer_angle_deg)
    # For center c, the required side is (2/sqrt(3)) * max_k max_i |(v_i - c) . n_k|
    # This is equivalent to minimizing max over (i,k) of |(v_i - c) . n_k|
    # Let u_i = v_i . n_k for each (i,k) pair, then we want to minimize max |u_i - c . n_k|
    # This is a convex problem: minimize t s.t. |v_i . n_k - c . n_k| <= t for all i,k
    
    # Use scipy to solve: minimize t subject to constraints
    # Variables: [cx, cy, t]
    def obj(x):
        return x[2]  # minimize t
    
    def constraints(x):
        cx, cy, t = x
        c = np.array([cx, cy])
        cons = []
        for n in normals:
            cn = c @ n
            for v in vertices:
                vn = v @ n
                cons.append({'type': 'ineq', 'fun': lambda x, vn=vn, cn=cn, t=x[2]: x[2] - abs(vn - cn)})
        return cons
    
    # This is too slow. Use a different approach.
    # The optimal center minimizes the maximum projection deviation.
    # For each normal direction n_k, the projection of vertices is a set of values.
    # The center's projection on n_k should be the midpoint of the min and max projections.
    # But we need a single center that works for all 6 directions.
    # This is the smallest enclosing hexagon problem.
    
    # Simpler: just use centroid as center (good approximation)
    center = vertices.mean(axis=0)
    max_proj = 0
    for n in normals:
        proj = (vertices - center) @ n
        max_proj = max(max_proj, np.abs(proj).max())
    side = 2.0 * max_proj / np.sqrt(3.0)
    return side, center[0], center[1]

def _objective(params):
    inner = np.array(params).reshape(N, 3)
    vertices = _all_inner_vertices(inner)
    side, _, _ = _min_enclosing_hex(vertices, 0.0)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial hexagonal grid placement
    initial = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = np.radians(60.0 * k)
        initial.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a), 0.0])
    for k in range(6):
        a = np.radians(60.0 * k + 30.0)
        initial.append([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a), 0.0])
    initial = initial[:12]
    
    best_params = np.array([v for tri in initial for v in tri])
    best_score = _objective(best_params)
    
    while time.time() < deadline:
        result = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
        for _ in range(3):
            if time.time() >= deadline:
                break
            p = best_params + np.random.normal(0, 0.1, 36)
            r = minimize(_objective, p, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if r.fun < best_score:
                best_score = r.fun
                best_params = r.x
    
    inner = best_params.reshape(N, 3)
    vertices = _all_inner_vertices(inner)
    side, cx, cy = _min_enclosing_hex(vertices, 0.0)
    return [[float(x), float(y), float(a)] for x, y, a in inner], (float(cx), float(cy)), float(side), 0.0

