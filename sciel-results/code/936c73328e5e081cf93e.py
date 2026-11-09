"""Packing 12 unit hexagons inside a regular hexagon using gradient optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + k * np.pi / 3.0),
                      cy + side * np.sin(a + k * np.pi / 3.0)] for k in range(6)])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    lx = dx * np.cos(-a) - dy * np.sin(-a)
    ly = dx * np.sin(-a) + dy * np.cos(-a)
    s = side
    sqrt3 = np.sqrt(3.0)
    for k in range(6):
        theta = np.radians(30) + k * np.pi / 3.0
        if lx * np.cos(theta) + ly * np.sin(theta) > s * sqrt3 / 2.0:
            return False
    return True

def min_separation(v1, v2):
    """Minimum distance between two convex polygons. Returns negative if overlapping."""
    # Use SAT to find minimum overlap or separation
    def get_edges(verts):
        n = len(verts)
        edges = []
        for i in range(n):
            edges.append(verts[(i+1) % n] - verts[i])
        return np.array(edges)
    
    e1 = get_edges(v1)
    e2 = get_edges(v2)
    
    # Get all edge normals
    def normals_from_edges(edges):
        norms = []
        for e in edges:
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                norms.append(n / norm)
        return np.array(norms)
    
    n1 = normals_from_edges(e1)
    n2 = normals_from_edges(e2)
    all_n = np.vstack([n1, n2])
    
    # For each axis, compute the gap
    min_gap = np.inf
    for n in all_n:
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap = max(proj1.min(), proj2.min()) - min(proj1.max(), proj2.max())
        # Actually: separation = max(min1, min2) - min(max1, max2)
        # If negative, they overlap on this axis
        sep = max(proj1.min(), proj2.min()) - min(proj1.max(), proj2.max())
        min_gap = min(min_gap, sep)
    
    return min_gap

def objective(params):
    t = time.time()
    if t - GLOBAL_START > DEADLINE:
        return 1e6
    
    inner = params[:N*3].reshape(N, 3)
    ox, oy, os, oa = params[N*3:]
    
    if os < 1.0:
        return 1e6
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
            v2 = hex_vertices(inner[j,0], inner[j,1], 1.0, inner[j,2])
            sep = min_separation(v1, v2)
            if sep < 0:
                overlap_pen += sep * sep
    
    # Containment penalty
    contain_pen = 0.0
    for i in range(N):
        verts = hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for vx, vy in verts:
            if not point_in_hex(vx, vy, ox, oy, os, oa):
                contain_pen += 1.0
    
    return os + 100.0 * overlap_pen + 100.0 * contain_pen

def construct():
    global GLOBAL_START
    GLOBAL_START = time.time()
    
    sqrt3 = np.sqrt(3.0)
    d = sqrt3
    
    positions = [[0.0, 0.0]]
    for k in range(6):
        angle = k * np.pi / 3.0
        positions.append([d * np.cos(angle), d * np.sin(angle)])
    for k in range(6):
        angle = k * np.pi / 3.0 + np.pi/6
        positions.append([2*d * np.cos(angle), 2*d * np.sin(angle)])
    
    positions = positions[:N]
    
    inner_init = []
    for x, y in positions:
        inner_init.append([x, y, 0.0])
    
    max_extent = max(np.sqrt(x**2 + y**2) for x, y in positions)
    outer_side_init = max_extent + 2.0
    outer_init = [0.0, 0.0, outer_side_init, 0.0]
    
    x0 = np.array([v for row in inner_init for v in row] + outer_init)
    
    bounds = []
    for i in range(N):
        bounds += [(-50, 50), (-50, 50), (-180, 180)]
    bounds += [(-50, 50), (-50, 50), (1.0, 20.0), (-180, 180)]
    
    best_result = None
    best_obj = np.inf
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 3000})
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if best_result is None:
        best_result = x0
    
    inner = best_result[:N*3].reshape(N, 3)
    ox, oy, os, oa = best_result[N*3:]
    
    inner_list = [[float(inner[i,0]), float(inner[i,1]), float(inner[i,2])] for i in range(N)]
    
    if os < 1.0:
        os = 1.0
    
    return inner_list, (float(ox), float(oy)), float(os), float(oa)

GLOBAL_START = 0.0
