import numpy as np
import time
from scipy.optimize import minimize
import math

N = 11
START_TIME = time.time()
TIME_LIMIT = 100.0 # Safety margin

def _hex_vertices(center, angle_deg, side=1.0):
    """
    Return vertices of a regular hexagon with given center, rotation, and side length.
    """
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    R = np.array([[c, -s], [s, c]])
    # Standard hexagon vertices (flat-top? pointy-top?)
    # Let's use pointy-top for standard orientation (vertex at top)
    # Angles: 30, 90, 150, 210, 270, 330 degrees?
    # Or 0, 60, 120...
    # Let's use 0, 60, 120, 180, 240, 300 (vertex at right)
    angles = np.deg2rad(np.arange(0, 360, 60))
    verts = np.column_stack([np.cos(angles), np.sin(angles)]) * side
    verts = verts @ R.T
    verts += np.array(center)
    return verts

def _point_in_poly(point, poly):
    """
    Check if point is inside a convex polygon (CCW or CW).
    Uses cross product sign consistency.
    """
    n = len(poly)
    sign = None
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = np.cross(p2 - p1, point - p1)
        if abs(cross) > 1e-9:
            if sign is None:
                sign = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """
    Minimum distance between two segments p1-p2 and p3-p4.
    """
    # Vector d1 = p2 - p1, d2 = p4 - p3
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    EPS = 1e-9
    if a <= EPS and e <= EPS:
        return np.linalg.norm(r)
    if a <= EPS:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(r + t * d2)
    
    c = np.dot(d1, r)
    if e <= EPS:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
        return np.linalg.norm(r + s * d1)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > EPS:
        s = (b * f - c * e) / denom
    else:
        s = 0.0
    
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - c) / a))
    else:
        if s < 0.0:
            s = 0.0
            t = max(0.0, min(1.0, f / e))
        elif s > 1.0:
            s = 1.0
            t = max(0.0, min(1.0, (b + f) / e))
    
    d1p = p1 + s * d1
    d2p = p3 + t * d2
    return np.linalg.norm(d1p - d2p)

def _min_dist_poly(poly1, poly2):
    """
    Minimum distance between two convex polygons.
    Returns 0 if they intersect or touch.
    """
    # Check vertex inclusion
    for v in poly1:
        if _point_in_poly(v, poly2):
            return 0.0
    for v in poly2:
        if _point_in_poly(v, poly1):
            return 0.0
            
    # Check edge intersections
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            # Simple segment intersection check
            d1 = p2 - p1
            d2 = p4 - p3
            r = p1 - p3
            denom = np.cross(d1, d2)
            if abs(denom) > 1e-9:
                t = np.cross(r, d2) / denom
                u = np.cross(r, d1) / denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return 0.0
            else:
                # Parallel or collinear
                if abs(np.cross(d1, r)) < 1e-9:
                    # Collinear
                    # Check if projections overlap
                    proj1 = np.dot(p1, d1)
                    proj2 = np.dot(p2, d1)
                    proj3 = np.dot(p3, d1)
                    proj4 = np.dot(p4, d1)
                    min1, max1 = min(proj1, proj2), max(proj1, proj2)
                    min2, max2 = min(proj3, proj4), max(proj3, proj4)
                    if max1 >= min2 and min1 <= max2:
                        return 0.0
    
    # Min distance between edges
    min_d = float('inf')
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_d:
                min_d = d
    return min_d

def _max_dist_point_poly(point, poly):
    """
    Maximum distance from a point to vertices of a polygon.
    Since the polygon is convex, the max distance is to a vertex.
    """
    return np.max(np.linalg.norm(poly - point, axis=1))

def construct():
    # Initialize with a known valid packing for S=5
    # Center at (0,0)
    # 6 neighbors at dist sqrt(3)
    # 4 from second ring at dist 2*sqrt(3)
    
    sqrt3 = math.sqrt(3)
    
    # Positions
    x = [0.0]
    y = [0.0]
    angles = [0.0]
    
    # Neighbors
    for k in range(6):
        angle = k * 60
        rad = math.radians(angle)
        x.append(sqrt3 * math.cos(rad))
        y.append(sqrt3 * math.sin(rad))
        angles.append(0.0)
        
    # Second ring (4 hexagons)
    # Axial (2,0) -> (2*sqrt3, 0)
    x.append(2 * sqrt3)
    y.append(0.0)
    angles.append(0.0)
    
    # Axial (1,1) -> (1.5*sqrt3, 1.5)
    x.append(1.5 * sqrt3)
    y.append(1.5)
    angles.append(0.0)
    
    # Axial (0,2) -> (sqrt3, 3)
    x.append(sqrt3)
    y.append(3.0)
    angles.append(0.0)
    
    # Axial (-1, 2) -> (-0.5*sqrt3, 3)
    x.append(-0.5 * sqrt3)
    y.append(3.0)
    angles.append(0.0)
    
    x = np.array(x)
    y = np.array(y)
    angles = np.array(angles)
    
    S = 5.0
    
    # Flatten parameters: x, y, angles, S
    params = np.concatenate([x, y, angles, [S]])
    
    def objective(p):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        
        x_i = p[0:N]
        y_i = p[N:2*N]
        ang_i = p[2*N:3*N]
        S_val = p[3*N]
        
        if S_val < 1.0:
            return 1e10
            
        penalty = 0.0
        
        # Compute vertices for all hexagons
        verts_list = []
        for i in range(N):
            verts_list.append(_hex_vertices((x_i[i], y_i[i]), ang_i[i]))
            
        # Containment penalty
        R_in = S_val * sqrt3 / 2.0
        for i in range(N):
            d_max = _max_dist_point_poly((0.0, 0.0), verts_list[i])
            if d_max > R_in:
                penalty += (d_max - R_in) ** 2
                
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                d = _min_dist_poly(verts_list[i], verts_list[j])
                if d < 0.01:
                    penalty += (0.01 - d) ** 2
                    
        return S_val + 100.0 * penalty
    
    # Optimize
    try:
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-5})
        
        if res.success or res.fun < 1e10:
            best_params = res.x
        else:
            best_params = params
    except Exception:
        best_params = params
        
    # Extract best
    x_best = best_params[0:N]
    y_best = best_params[N:2*N]
    ang_best = best_params[2*N:3*N]
    S_best = best_params[3*N]
    
    # Ensure S_best is not too small (invalid)
    if S_best < 3.0:
        S_best = 5.0
        x_best = x
        y_best = y
        ang_best = angles
        
    inner = [[x_best[i], y_best[i], ang_best[i]] for i in range(N)]
    return inner, (0.0, 0.0), S_best, 0.0

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END
