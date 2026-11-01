"""Packing 12 unit hexagons in a minimal regular hexagon using force-directed simulation."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.radians(60 * i)
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon using half-plane tests."""
    angle = np.radians(angle_deg)
    for i in range(6):
        edge_angle = angle + np.radians(60 * i + 30)
        nx = np.cos(edge_angle)
        ny = np.sin(edge_angle)
        d = (px - cx) * nx + (py - cy) * ny
        if d > side * np.sqrt(3) / 2:
            return False
    return True

def _hexagon_distance(h1, h2):
    """Approximate minimum distance between two hexagons."""
    cx1, cy1, _ = h1
    cx2, cy2, _ = h2
    dx = cx2 - cx1
    dy = cy2 - cy1
    center_dist = np.sqrt(dx*dx + dy*dy)
    return center_dist - 2.0

def _compute_bounding_hexagon(inner):
    """Compute minimum enclosing regular hexagon for the set of hexagons."""
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0
    
    all_points = []
    for ix, iy, ia in inner:
        verts = _hexagon_vertices(ix, iy, 1.0, ia)
        all_points.extend(verts)
    all_points = np.array(all_points)
    
    for outer_angle in range(0, 60, 3):
        oa = np.radians(outer_angle)
        max_extent = 0
        cx_sum = 0
        cy_sum = 0
        for k in range(3):
            edge_angle = oa + np.radians(30 + 60 * k)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            projections = all_points[:, 0] * nx + all_points[:, 1] * ny
            min_p = np.min(projections)
            max_p = np.max(projections)
            extent = (max_p - min_p) / np.sqrt(3)
            max_extent = max(max_extent, extent)
            cx_sum += (min_p + max_p) / 2 / 3
            cy_sum += (min_p + max_p) / 2 / 3
        
        if max_extent < best_side:
            best_side = max_extent
            best_center = (cx_sum, cy_sum)
            best_angle = outer_angle
    
    return best_center, best_side, best_angle

def _objective(params):
    """Objective function: minimize outer hexagon side length with overlap penalty."""
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        inner.append([x, y, angle])
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _hexagon_distance(inner[i], inner[j])
            if d < 0.05:
                penalty += (0.05 - d) ** 2
    
    _, side, _ = _compute_bounding_hexagon(inner)
    
    return side + 100 * penalty

# EVOLVE-BLOCK-START
def construct():
    """Construct 12 unit hexagons in a minimal regular hexagon."""
    np.random.seed(42)
    
    params = np.zeros(3 * N)
    for i in range(N):
        if i == 0:
            params[3*i] = 0.0
            params[3*i+1] = 0.0
        elif i <= 6:
            k = i - 1
            angle = np.radians(60 * k)
            params[3*i] = 2.0 * np.cos(angle)
            params[3*i+1] = 2.0 * np.sin(angle)
        else:
            k = i - 7
            angle = np.radians(60 * k + 30)
            params[3*i] = 3.5 * np.cos(angle)
            params[3*i+1] = 3.5 * np.sin(angle)
        params[3*i+2] = 0.0
    
    result = minimize(_objective, params, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
    
    inner = []
    for i in range(N):
        x = result.x[3*i]
        y = result.x[3*i+1]
        angle = result.x[3*i+2]
        inner.append([x, y, angle])
    
    center, side, angle = _compute_bounding_hexagon(inner)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
