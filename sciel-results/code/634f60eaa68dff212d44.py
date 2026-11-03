"""Rewritten program for packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        vertices.append((x, y))
    return vertices

def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segment_intersection(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect (including one containing the other)."""
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return True
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return True
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersection(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                return True
    return False

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate the minimum distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)

def distance_point_to_polygon(px, py, polygon):
    """Calculate the minimum distance from point to polygon boundary. Returns 0 if inside."""
    if point_in_polygon(px, py, polygon):
        return 0.0
    
    min_dist = float('inf')
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = distance_point_to_segment(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist

MIN_DIST_TABLE = np.zeros(61)

def init_min_dist_table():
    """
    Precompute the minimum center distance for two unit hexagons 
    to avoid overlap for relative angles 0 to 60 degrees.
    """
    for i in range(61):
        rel_angle = i * 1.0
        lo = 0.0
        hi = 2.0
        for _ in range(20):
            mid = (lo + hi) / 2
            hex1 = hexagon_vertices(0, 0, 1.0, 0.0)
            hex2 = hexagon_vertices(mid, 0, 1.0, rel_angle)
            
            intersect = False
            for v in hex1:
                if point_in_polygon(v[0], v[1], hex2):
                    intersect = True
                    break
            if not intersect:
                for v in hex2:
                    if point_in_polygon(v[0], v[1], hex1):
                        intersect = True
                        break
            
            if not intersect:
                n1 = len(hex1)
                n2 = len(hex2)
                for i1 in range(n1):
                    for i2 in range(n2):
                        p1 = hex1[i1]
                        p2 = hex1[(i1+1)%n1]
                        p3 = hex2[i2]
                        p4 = hex2[(i2+1)%n2]
                        if segment_intersection(p1, p2, p3, p4):
                            intersect = True
                            break
                    if intersect: break
            
            if intersect:
                lo = mid
            else:
                hi = mid
        
        MIN_DIST_TABLE[i] = hi

def get_min_dist(angle1, angle2):
    """Get minimum center distance from table."""
    d_angle = abs(angle1 - angle2) % 60.0
    idx = int(round(d_angle))
    return MIN_DIST_TABLE[idx]

def construct():
    """
    Construct 11 unit hexagons packed in a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
    """
    N_local = 11
    init_min_dist_table()
    
    sqrt3 = math.sqrt(3)
    
    init = []
    init.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = i * math.pi / 3
        init.append((sqrt3 * math.cos(a), sqrt3 * math.sin(a), 0.0))
    for i in range(4):
        a = (i + 0.5) * math.pi / 3
        init.append((2*sqrt3 * math.cos(a), 2*sqrt3 * math.sin(a), 0.0))
        
    init[6] = (sqrt3 * math.cos(math.pi/3), sqrt3 * math.sin(math.pi/3), 30.0)

    def objective(params):
        S = params[-1]
        if S < 2.0:
            return 1000.0
            
        outer_poly = hexagon_vertices(0.0, 0.0, S, 0.0)
        
        penalty = 0.0
        
        centers = []
        angles = []
        for i in range(N_local):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            centers.append((x, y))
            angles.append(a)
            
            hex_i = hexagon_vertices(x, y, 1.0, a)
            for v in hex_i:
                dist = distance_point_to_polygon(v[0], v[1], outer_poly)
                if dist > 0:
                    penalty += dist * dist * 10.0
        
        for i in range(N_local):
            for j in range(i+1, N_local):
                d_req = get_min_dist(angles[i], angles[j])
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_actual = math.hypot(dx, dy)
                if d_actual < d_req:
                    penalty += (d_req - d_actual) ** 2 * 10.0
        
        return S + penalty

    x0 = np.zeros(3 * N_local + 1)
    for i in range(N_local):
        x0[3*i] = init[i][0]
        x0[3*i+1] = init[i][1]
        x0[3*i+2] = init[i][2]
    x0[-1] = 5.0
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6})
    
    best_params = result.x
    best_val = result.fun
    
    np.random.seed(42)
    for _ in range(3):
        x0_perturbed = x0.copy()
        x0_perturbed[:3*N_local] += np.random.normal(0, 0.5, size=3*N_local)
        x0_perturbed[-1] = 5.0 + np.random.normal(0, 0.5)
        
        res_pert = minimize(objective, x0_perturbed, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
        if res_pert.fun < best_val:
            best_val = res_pert.fun
            best_params = res_pert.x

    inner = []
    for i in range(N_local):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    outer_side = best_params[-1]
    
    outer_poly = hexagon_vertices(0.0, 0.0, outer_side, 0.0)
    inner_polys = []
    for i in range(N_local):
        poly = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        inner_polys.append(poly)
        if not all(point_in_polygon(v[0], v[1], outer_poly) for v in poly):
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0
            
    for i in range(N_local):
        for j in range(i+1, N_local):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return [[4.0 * k, 0.0, 0.0] for k in range(-(N_local // 2), N_local - N_local // 2)], (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), outer_side, 0.0

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
