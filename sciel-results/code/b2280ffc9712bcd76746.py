"""Optimal transport-inspired hexagon packing for n=12."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def polygon_area(verts):
    """Shoelace formula for polygon area."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return abs(area) / 2.0

def point_in_polygon(px, py, verts):
    """Ray casting point-in-polygon test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects with segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def polygons_intersect(verts1, verts2):
    """Check if two convex polygons intersect (edges cross or one contains a vertex of the other)."""
    # Check if any vertex of one is inside the other
    for v in verts1:
        if point_in_polygon(v[0], v[1], verts2):
            return True
    for v in verts2:
        if point_in_polygon(v[0], v[1], verts1):
            return True
    # Check if any edges intersect
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            l = (k + 1) % n2
            if segments_intersect(verts1[i], verts1[j], verts2[k], verts2[l]):
                return True
    return False

def outer_hex_contains_point(px, py, outer_center, outer_side, outer_angle_deg):
    """Check if point is inside the outer hexagon."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle_deg)
    return point_in_polygon(px, py, outer_verts)

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg, tol=1e-6):
    """
    Compute the minimum side length of the outer hexagon (with given center and rotation)
    that contains all inner hexagons.
    For a regular hexagon with center c, side s, and rotation angle a, a point p is inside
    if for each of the 6 face normals, the projection of (p-c) onto the normal <= s * sqrt(3)/2.
    """
    cx, cy = outer_center
    angle_rad = math.radians(outer_angle_deg)
    
    # The 6 face normals for a regular hexagon
    # For a regular hexagon with "pointy top" orientation, faces are at angles 0, 60, 120, 180, 240, 300
    # The apothem (distance from center to face midpoint) = side * sqrt(3) / 2
    # A point is inside if its projection onto each face normal <= apothem
    
    max_required_apothem = 0.0
    
    for hex_data in inner_hexes:
        hx, hy, hangle = hex_data
        hverts = hex_vertices(hx, hy, 1.0, hangle)
        for vx, vy in hverts:
            dx = vx - cx
            dy = vy - cy
            for i in range(6):
                theta = angle_rad + i * math.pi / 3
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = abs(dx * nx + dy * ny)
                if proj > max_required_apothem:
                    max_required_apothem = proj
    
    # apothem = side * sqrt(3) / 2 => side = apothem * 2 / sqrt(3)
    return max_required_apothem * 2.0 / math.sqrt(3.0)

def objective(params):
    """Objective function: outer side length. params = [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12]"""
    inner_hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, a))
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Check for overlaps and compute outer side
    # First, compute the outer side needed
    outer_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle)
    
    # Check overlaps
    all_verts = [hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner_hexes]
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(all_verts[i], all_verts[j]):
                # Penalty for overlap
                # Find minimum distance to separate - use a large penalty
                # Compute approximate overlap penalty based on center distance
                dx = inner_hexes[i][0] - inner_hexes[j][0]
                dy = inner_hexes[i][1] - inner_hexes[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlapping unit hexagons is roughly 2*apothem = sqrt(3)
                min_dist = math.sqrt(3.0)
                if dist < min_dist:
                    penalty = (min_dist - dist) * 100.0
                else:
                    penalty = 0.0
                return outer_side + penalty
    
    return outer_side

def construct():
    # Initial positions: spread out in a ring pattern
    # Use optimal transport-inspired initialization: place hexagons at vertices and center of a larger hexagonal arrangement
    # For 12 hexagons, place them at positions that maximize spacing
    
    # Ring of 6 + inner ring of 6
    initial_params = []
    
    # Outer ring: 6 hexagons at radius ~3, angles 0, 60, 120, 180, 240, 300
    r_outer = 3.0
    for i in range(6):
        angle = i * math.pi / 3
        x = r_outer * math.cos(angle)
        y = r_outer * math.sin(angle)
        initial_params.extend([x, y, 0.0])
    
    # Inner ring: 6 hexagons at radius ~1.5, angles 30, 90, 150, 210, 270, 330
    r_inner = 1.5
    for i in range(6):
        angle = (i * math.pi / 3) + math.pi / 6
        x = r_inner * math.cos(angle)
        y = r_inner * math.sin(angle)
        initial_params.extend([x, y, 0.0])
    
    # Optimize
    best_params = list(initial_params)
    best_score = objective(best_params)
    
    # Try multiple random restarts with different initial configurations
    np.random.seed(42)
    for restart in range(20):
        if restart == 0:
            params = list(initial_params)
        else:
            # Random perturbation around the initial configuration
            params = []
            for i in range(N):
                if i < 6:
                    angle = i * math.pi / 3
                    r = r_outer + np.random.uniform(-0.5, 0.5)
                    x = r * math.cos(angle + np.random.uniform(-0.3, 0.3))
                    y = r * math.sin(angle + np.random.uniform(-0.3, 0.3))
                else:
                    angle = ((i-6) * math.pi / 3) + math.pi / 6
                    r = r_inner + np.random.uniform(-0.5, 0.5)
                    x = r * math.cos(angle + np.random.uniform(-0.3, 0.3))
                    y = r * math.sin(angle + np.random.uniform(-0.3, 0.3))
                params.extend([x, y, 0.0])
        
        result = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = list(result.x)
    
    # Extract inner hexagons from best params
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    outer_center = (0.0, 0.0)
    outer_side = best_score
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
